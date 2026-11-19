# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
from transformers import LlamaConfig

import vllm.config.parallel as parallel_config_module
from vllm.config import ModelConfig, ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.config.parallel import EPLBConfig
from vllm.config.utils import hash_config_cached
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.distributed.eplb.async_worker import run_rebalance_experts
from vllm.distributed.eplb.eplb_communicator import TorchDistGlooStagedEplbCommunicator
from vllm.distributed.eplb.eplb_state import (
    EplbModelState,
    EplbState,
    EplbStats,
    compute_logical_maps,
)
from vllm.distributed.eplb.eplb_utils import CpuGpuEvent
from vllm.distributed.eplb.policy import DefaultEplbPolicy, TemporalEplbStats
from vllm.distributed.eplb.policy.stair import StairConfig, StairPlanner
from vllm.distributed.eplb.policy.stair.runtime import StairEplbPolicy
from vllm.distributed.eplb.rebalance_execute import AsyncEplbLayerResult, transfer_layer
from vllm.distributed.parallel_state import get_ep_group
from vllm.platforms import current_platform
from vllm.platforms.interface import Platform, PlatformEnum


class _Platform(Platform):
    _enum = PlatformEnum.OOT

    @classmethod
    def supports_eplb(cls):
        return True

    @classmethod
    def check_and_update_eplb_config(cls, parallel_config):
        config = parallel_config.eplb_config
        if config.policy is None:
            config.policy = "custom"
        if config.policy != "custom" or config.communicator not in (
            None,
            "custom_comm",
        ):
            raise ValueError("unsupported custom EPLB configuration")
        config.communicator = "custom_comm"


@pytest.mark.parametrize(
    "policy, communicator", [(None, None), ("custom", None), ("custom", "custom_comm")]
)
def test_platform_can_validate_out_of_tree_names(monkeypatch, policy, communicator):
    monkeypatch.setattr(parallel_config_module, "current_platform", _Platform())
    config = ParallelConfig(
        enable_eplb=True,
        enable_expert_parallel=True,
        tensor_parallel_size=2,
        distributed_executor_backend="uni",
        eplb_config=EPLBConfig(policy=policy, communicator=communicator),
    )
    assert config.eplb_config.policy == "custom"
    assert config.eplb_config.communicator == "custom_comm"


@pytest.mark.parametrize("field_name", ["policy", "communicator"])
def test_builtin_platform_rejects_unknown_names(field_name):
    config = ParallelConfig(eplb_config=EPLBConfig(**{field_name: "unknown"}))
    with pytest.raises(ValueError, match=f"Unknown EPLB {field_name}"):
        Platform.check_and_update_eplb_config(config)


def test_platform_can_reject_eplb_even_on_cuda(monkeypatch):
    class UnsupportedCuda(Platform):
        _enum = PlatformEnum.CUDA

        @classmethod
        def supports_eplb(cls):
            return False

    monkeypatch.setattr(parallel_config_module, "current_platform", UnsupportedCuda())
    with pytest.raises(ValueError, match="not supported on this platform"):
        ParallelConfig(
            enable_eplb=True,
            tensor_parallel_size=2,
            distributed_executor_backend="uni",
        )


def test_xpu_preserves_eplb_and_xccl_auto_selection():
    class XpuPlatform(Platform):
        _enum = PlatformEnum.XPU

    config = ParallelConfig()
    assert XpuPlatform.supports_eplb()
    XpuPlatform.check_and_update_eplb_config(config)
    assert config.eplb_config.communicator == "torch_xccl"
    assert config.eplb_config.policy == "default"


@pytest.mark.parametrize(
    "communicator, use_async",
    [
        ("torch_gloo", True),
        ("torch_xccl", True),
        ("nixl", True),
        ("torch_nccl", False),
        ("pynccl", False),
    ],
)
def test_explicit_builtin_communicator_is_preserved(communicator, use_async):
    config = ParallelConfig(
        eplb_config=EPLBConfig(communicator=communicator, use_async=use_async)
    )
    Platform.check_and_update_eplb_config(config)
    assert config.eplb_config.communicator == communicator


@pytest.mark.parametrize("communicator", ["torch_nccl", "pynccl"])
def test_async_nccl_restrictions_remain(communicator):
    config = ParallelConfig(eplb_config=EPLBConfig(communicator=communicator))
    with pytest.raises(ValueError, match="incompatible with async EPLB"):
        Platform.check_and_update_eplb_config(config)


def test_llm_rejects_async_nccl_communicator(vllm_runner, monkeypatch, tmp_path):
    """Reject an invalid async communicator before loading model weights."""

    class BuiltinPlatform(Platform):
        _enum = PlatformEnum.CUDA

    monkeypatch.setattr(parallel_config_module, "current_platform", BuiltinPlatform())
    LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    ).save_pretrained(tmp_path)
    eplb_config = EPLBConfig(communicator="torch_nccl")
    with pytest.raises(ValueError, match="incompatible with async EPLB"):
        vllm_runner(
            str(tmp_path),
            max_model_len=32,
            skip_tokenizer_init=True,
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            enable_eplb=True,
            distributed_executor_backend="uni",
            eplb_config=eplb_config,
        )


@pytest.mark.parametrize("interval", [0, -1])
def test_balancedness_interval_must_be_positive(interval):
    with pytest.raises(ValueError, match="log_balancedness_interval"):
        EPLBConfig(log_balancedness_interval=interval)


@pytest.fixture
def cpu_group(tmp_path):
    config = VllmConfig()
    # Build the real foreground and background groups for CPU component tests.
    config.parallel_config.enable_eplb = True
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"file://{tmp_path}/store",
            backend="gloo",
        )
        initialize_model_parallel(backend="gloo")
        try:
            yield get_ep_group().cpu_group
        finally:
            cleanup_dist_env_and_memory()


class _ExpertModel(torch.nn.Module):
    """Small real expert weight owner for control-plane and transfer checks."""

    def __init__(self, num_experts=4, ep_size=1):
        super().__init__()
        num_local = num_experts // ep_size
        self.weight = torch.nn.Parameter(
            torch.arange(num_local * 4).reshape(num_local, 2, 2).float(), False
        )
        self.expert_weights = [[self.weight]]
        self.num_moe_layers = 1
        self.num_routed_experts = self.num_logical_experts = num_experts
        self.num_redundant_experts = 0
        self.num_physical_experts = num_experts
        self.num_local_physical_experts = num_local
        self.num_expert_groups = 1
        self.moe_layers = ()

    def set_eplb_state(self, loads, logical_map, replica_counts):
        self.load_buffer = loads


class _ModelState(EplbModelState):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_commit = False
        self.committed_layers = []
        self.committed_maps = []

    def _commit_eplb_maps_for_layer(self, target, layer):
        super()._commit_eplb_maps_for_layer(target, layer)
        assert not self.pending_result.consumed_event._recorded.is_set()
        if self.fail_commit:
            raise RuntimeError("device commit failed")
        self.committed_layers.append(layer)
        self.committed_maps.append(self.physical_to_logical_map.clone())

    def _commit_eplb_maps(self, new_physical_to_logical_map):
        super()._commit_eplb_maps(new_physical_to_logical_map)
        self.committed_layers.extend(range(self.model.num_moe_layers))
        self.committed_maps.append(self.physical_to_logical_map.clone())


class _State(EplbState):
    model_state_cls = _ModelState

    def __init__(self, policy=None, parallel_config=None):
        self.selected_policy = policy or DefaultEplbPolicy()
        super().__init__(parallel_config or ParallelConfig(), torch.device("cpu"))

    def create_policy(self):
        return self.selected_policy


@pytest.fixture
def model_state(cpu_group):
    model = _ExpertModel()
    mapping = torch.arange(4).reshape(1, 4)
    logical, replicas = compute_logical_maps(mapping, 4)
    loads = torch.zeros(1, 4, dtype=torch.int64)
    return _State.model_state_cls(
        physical_to_logical_map=mapping,
        physical_to_logical_map_buffer=mapping,
        logical_to_physical_map=logical,
        logical_replica_count=replicas,
        expert_load_pass=loads,
        expert_load_window=torch.zeros(4, 1, 4, dtype=torch.int64),
        model_name="expert-test",
        model=model,
        expert_buffer=[torch.empty_like(model.weight)],
        rebalanced=False,
        eplb_stats=None,
        device_index=None,
        communicator=TorchDistGlooStagedEplbCommunicator(cpu_group),
    )


@pytest.mark.parametrize("use_async", [False, True])
def test_policy_receives_extended_stats_without_losing_inputs(model_state, use_async):
    @dataclass
    class Stats(EplbStats):
        metadata: tuple[int, ...] = ()

    class Policy(DefaultEplbPolicy):
        def rebalance_experts(self, stats, old_global_expert_indices=None):
            assert stats is published_stats
            assert isinstance(stats, Stats)
            assert stats.metadata == (3, 7)
            assert stats.global_expert_load_window.device.type == "cpu"
            torch.testing.assert_close(stats.global_expert_load_window, loads)
            return super().rebalance_experts(stats, old_global_expert_indices)

    loads = torch.tensor([[13, 3, 4, 3]])
    current = torch.tensor([[0, 1, 2, 3, 0, 1]])
    device = "cpu"
    if use_async and (current_platform.is_cuda_alike() or current_platform.is_xpu()):
        device = current_platform.device_type
    published_stats = Stats(loads.to(device), 6, 1, 1, 2, metadata=(3, 7))
    state = _State(Policy())
    if use_async:
        model_state.eplb_stats = published_stats
        result = run_rebalance_experts(model_state, state, current, None)
    else:
        result = state.policy.rebalance_experts(published_stats, current)
    default_state = EplbState(ParallelConfig(), torch.device("cpu"))
    assert isinstance(default_state.policy, DefaultEplbPolicy)
    expected = default_state.policy.rebalance_experts(published_stats, current)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("fail_commit", [False, True])
def test_async_commit_updates_maps_before_buffer_acknowledgement(
    model_state, fail_commit
):
    model_state.fail_commit = fail_commit
    target = torch.tensor([[1, 0, 3, 2]])
    original = model_state.model.weight.detach().clone()
    metadata = transfer_layer(
        old_layer_indices=model_state.physical_to_logical_map[0],
        new_layer_indices=target[0],
        expert_weights=model_state.model.expert_weights[0],
        expert_weights_buffer=model_state.expert_buffer,
        communicator=model_state.communicator,
        ep_group=get_ep_group().device_group,
        stream=None,
        is_profile=False,
    )
    result = AsyncEplbLayerResult(
        0, target[0], metadata, CpuGpuEvent(), predicted_mean_ratio=1.05
    )
    model_state.pending_result = result
    model_state.rebalanced = True
    if fail_commit:
        with pytest.raises(RuntimeError, match="commit failed"):
            model_state._move_to_workspace(0)
    else:
        model_state._move_to_workspace(0)
    torch.testing.assert_close(model_state.model.weight, original[[1, 0, 3, 2]])
    torch.testing.assert_close(model_state.physical_to_logical_map, target)
    if fail_commit:
        assert model_state.committed_layers == []
        assert model_state.last_committed_mean_ratios is None
        assert model_state.pending_result is result
        assert not result.consumed_event._recorded.is_set()
    else:
        assert model_state.committed_layers == [0]
        np.testing.assert_array_equal(model_state.last_committed_mean_ratios, [1.05])
        torch.testing.assert_close(model_state.committed_maps[0], target)
        assert model_state.pending_result is None
        assert result.consumed_event._recorded.is_set()


def test_sync_commit_updates_maps_before_device_work(model_state):
    target = torch.tensor([[1, 0, 3, 2]])
    model_state._commit_eplb_maps(target)
    assert model_state.committed_layers == [0]
    torch.testing.assert_close(model_state.committed_maps[0], target)
    torch.testing.assert_close(
        model_state.logical_to_physical_map[0, :, 0], torch.tensor([1, 0, 3, 2])
    )


class _TemporalPolicy(DefaultEplbPolicy):
    requires_temporal_load = True

    def rebalance_experts(self, stats, old_global_expert_indices=None):
        assert isinstance(stats, TemporalEplbStats)
        assert old_global_expert_indices is not None
        return old_global_expert_indices


@pytest.fixture
def temporal_model(tmp_path, cpu_group):
    LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    ).save_pretrained(tmp_path)
    config = ModelConfig(model=str(tmp_path), skip_tokenizer_init=True)
    return config, _ExpertModel()


@pytest.mark.parametrize("use_async", [False, True])
def test_temporal_window_preserves_iteration_order_and_valid_zero_samples(
    temporal_model, use_async
):
    """Dummy, phase-excluded and missing forwards cannot masquerade as zero load."""
    config, model = temporal_model
    state = _State(
        _TemporalPolicy(),
        ParallelConfig(
            eplb_config=EPLBConfig(
                window_size=4,
                step_interval=4,
                load_collection_phase="decode",
                communicator="torch_gloo",
                use_async=use_async,
            )
        ),
    )
    state.add_model(model, config)
    published = None
    for step in range(7):
        state.set_batch_phase(batch_has_prefill=step == 2)
        if step != 5:
            state.prepare_forward(config, num_unpadded_tokens=1)
            model.load_buffer.fill_(0 if step == 3 else step + 1)
        state.step(is_dummy=step == 1)
        if use_async and step == 0:
            published = next(iter(state.model_states.values())).eplb_stats
    stats = next(iter(state.collect_temporal_load_stats().values()))
    assert stats.sample_counts == (1, 1, 1)
    assert stats.rank_node_ids == (0,)
    torch.testing.assert_close(
        stats.global_expert_load_window[:, 0, 0],
        torch.tensor([0, 5, 7], dtype=torch.int32),
    )
    if use_async:
        assert isinstance(published, TemporalEplbStats)
        torch.testing.assert_close(
            published.global_expert_load_window, torch.ones(1, 1, 4, dtype=torch.int32)
        )


def _temporal_load_worker(rank, world_size, directory):
    parallel = ParallelConfig(
        tensor_parallel_size=world_size,
        distributed_executor_backend="mp",
        eplb_config=EPLBConfig(
            window_size=4,
            step_interval=4,
            load_collection_phase="decode",
            communicator="torch_gloo",
            use_async=False,
        ),
    )
    config = VllmConfig(parallel_config=parallel)
    parallel.enable_eplb = True
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"file://{directory}/store",
            backend="gloo",
        )
        initialize_model_parallel(tensor_model_parallel_size=world_size, backend="gloo")
        try:
            state = _State(_TemporalPolicy(), parallel)
            target = _ExpertModel(ep_size=world_size)
            draft = _ExpertModel(num_experts=6, ep_size=world_size)
            target_config = ModelConfig(
                model=f"{directory}/target", skip_tokenizer_init=True
            )
            draft_config = ModelConfig(
                model=f"{directory}/draft", skip_tokenizer_init=True
            )
            state.add_model(target, target_config)
            state.add_model(draft, draft_config)
            for step in range(7):
                state.set_batch_phase(batch_has_prefill=rank == 0 and step == 2)
                if (rank == 0 and step != 5) or (rank == 1 and step % 2):
                    state.prepare_forward(target_config, num_unpadded_tokens=1)
                    target.load_buffer.fill_(
                        0 if step == 3 else (step + 1) * (1 if rank == 0 else 10)
                    )
                if rank == 1 and step == 5:
                    state.prepare_forward(draft_config, num_unpadded_tokens=1)
                    draft.load_buffer.fill_(100)
                state.step(is_dummy=rank == 1 and step == 1)
            target_stats, draft_stats = state.collect_temporal_load_stats().values()
            torch.testing.assert_close(
                target_stats.global_expert_load_window[:, 0, 0],
                torch.tensor([0, 5, 60, 7], dtype=torch.int32),
            )
            torch.testing.assert_close(
                draft_stats.global_expert_load_window,
                torch.full((1, 1, 6), 100, dtype=torch.int32),
            )
            assert target_stats.sample_counts == (1, 1, 1, 1)
            assert draft_stats.sample_counts == (1,)
            assert target_stats.rank_node_ids == draft_stats.rank_node_ids == (0, 0)
        finally:
            cleanup_dist_env_and_memory()


@pytest.mark.distributed
def test_temporal_reduction_aligns_rank_and_model_specific_activity(tmp_path):
    """Real Gloo ranks reduce matching time coordinates for heterogeneous models."""
    for name in ("target", "draft"):
        LlamaConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        ).save_pretrained(tmp_path / name)
    mp.spawn(_temporal_load_worker, args=(2, str(tmp_path)), nprocs=2, join=True)


def test_stair_subprocess_preserves_plans_and_reports_errors(cpu_group):
    """Real IPC preserves the CPU result and propagates planner input errors."""
    loads = np.tile([[[11, 11, 10, 8], [11, 11, 10, 8]]], (4, 1, 1))
    current = np.array([[[0, 1], [2, 3]]] * 2)
    history = np.full(2, np.nan)
    nodes = np.array([0, 0])
    config = StairConfig(load_window_bins=2)
    expected = StairPlanner.plan_rebalance(loads, current, history, nodes, config)
    policy = StairEplbPolicy(config)
    try:
        with pytest.raises(ValueError, match="non-negative"):
            policy.plan_sharded_rebalance(-loads, current, history, nodes, cpu_group)
        actual = policy.plan_sharded_rebalance(
            loads, current, history, nodes, cpu_group
        )
        for name in (
            "rank_expert_ids",
            "source_rank_ids",
            "source_slot_ids",
            "predicted_mean_ratios",
            "imbalance_ratios",
        ):
            np.testing.assert_array_equal(
                getattr(actual, name), getattr(expected, name)
            )
    finally:
        policy.close()
    with pytest.raises(RuntimeError, match="closed"):
        policy.plan_sharded_rebalance(loads, current, history, nodes, cpu_group)


def test_stair_does_not_restart_a_failed_planner(cpu_group):
    """A dead planner must fail the engine operation instead of silently retrying."""
    policy = StairEplbPolicy()
    args = (
        np.ones((2, 1, 2)),
        np.array([[[0], [1]]]),
        np.array([np.nan]),
        np.array([0, 0]),
        cpu_group,
    )
    try:
        policy.plan_sharded_rebalance(*args)
        process = policy._process
        assert process is not None
        process.kill()
        process.wait()
        with pytest.raises(RuntimeError, match="exited unexpectedly"):
            policy.plan_sharded_rebalance(*args)
        with pytest.raises(RuntimeError, match="closed"):
            policy.plan_sharded_rebalance(*args)
    finally:
        policy.close()


def test_stair_profile_keeps_weights_without_starting_planner(temporal_model):
    config, model = temporal_model
    state = EplbState(
        ParallelConfig(
            eplb_config=EPLBConfig(
                policy="stair", use_async=False, communicator="torch_nccl"
            )
        ),
        torch.device("cpu"),
    )
    state.add_model(model, config)
    original = model.weight.detach().clone()
    try:
        state.rearrange(is_profile=True)
        torch.testing.assert_close(model.weight, original, rtol=0, atol=0)
        assert isinstance(state.policy, StairEplbPolicy)
        assert state.policy._process is None
    finally:
        state.stop_async_loop()


def test_stair_rejects_aggregate_statistics():
    policy = StairEplbPolicy()
    try:
        with pytest.raises(ValueError, match="raw temporal"):
            policy.rebalance_experts(
                EplbStats(torch.ones(1, 4), 4, 1, 1, 2), torch.arange(4).reshape(1, 4)
            )
    finally:
        policy.close()


@pytest.mark.parametrize("field", ["enable_elastic_ep", "enable_fault_tolerance"])
def test_stair_rejects_unsupported_recovery(field):
    config = ParallelConfig(eplb_config=EPLBConfig(policy="stair"))
    setattr(config, field, True)
    with pytest.raises(ValueError, match="STAIR"):
        Platform.check_and_update_eplb_config(config)
    with pytest.raises(ValueError, match="STAIR"):
        EplbState(config, torch.device("cpu"))


def _stair_runtime_worker(rank, directory):
    parallel = ParallelConfig(
        tensor_parallel_size=2,
        distributed_executor_backend="mp",
        eplb_config=EPLBConfig(
            window_size=4,
            step_interval=16,
            communicator="torch_nccl",
            use_async=False,
            policy="stair",
            stair_config=StairConfig(load_window_bins=2),
        ),
    )
    config = VllmConfig(parallel_config=parallel)
    parallel.enable_eplb = True
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=2,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"file://{directory}/store",
            backend="gloo",
        )
        initialize_model_parallel(tensor_model_parallel_size=2, backend="gloo")
        state = EplbState(parallel, torch.device("cpu"))
        try:
            model = _ExpertModel(ep_size=2)
            model.weight.data.add_(rank * model.weight.numel())
            model_config = ModelConfig(model=directory, skip_tokenizer_init=True)
            state.add_model(model, model_config)
            for _ in range(4):
                state.prepare_forward(model_config, num_unpadded_tokens=1)
                model.load_buffer.copy_(
                    torch.tensor([[11, 11, 0, 0] if rank == 0 else [0, 0, 10, 8]])
                )
                state.step()
            model_state = state.model_states[hash_config_cached(model_config)]
            torch.testing.assert_close(
                model_state.physical_to_logical_map, torch.tensor([[0, 2, 1, 3]])
            )
            original = torch.arange(16).reshape(4, 2, 2).float()
            torch.testing.assert_close(
                model.weight, original[[0, 2] if rank == 0 else [1, 3]]
            )
            np.testing.assert_allclose(
                model_state.last_committed_mean_ratios, [1.05], rtol=0, atol=1e-15
            )
            stats = next(iter(state.collect_temporal_load_stats().values()))
            assert not np.shares_memory(
                stats.last_committed_mean_ratios, model_state.last_committed_mean_ratios
            )
            if rank == 0:
                stats.global_expert_load_window[0, 0, 0] = -1
            with pytest.raises(
                (ValueError, RuntimeError), match="non-negative|another"
            ):
                state.policy.rebalance_experts(
                    stats, model_state.physical_to_logical_map
                )
        finally:
            state.stop_async_loop()
            assert isinstance(state.policy, StairEplbPolicy)
            assert state.policy._process is None
            cleanup_dist_env_and_memory()


@pytest.mark.distributed
def test_stair_runtime_transfers_weights_and_commits_history(tmp_path):
    """Raw statistics, process shards, explicit sources and real Gloo agree."""
    LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    ).save_pretrained(tmp_path)
    mp.spawn(_stair_runtime_worker, args=(str(tmp_path),), nprocs=2, join=True)
