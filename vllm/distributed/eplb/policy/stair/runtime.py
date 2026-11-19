# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Sharded CPU planning and lifecycle for the STAIR EPLB policy."""

import os
import socket
import subprocess
import sys
import tempfile
from dataclasses import asdict
from multiprocessing.connection import Connection
from typing import BinaryIO

import numpy as np
import torch
from torch.distributed import ProcessGroup

from vllm.config.eplb import StairConfig
from vllm.distributed.eplb.transfer_plan import ExpertTransferPlan
from vllm.distributed.parallel_state import get_eplb_group

from ..abstract import AbstractEplbPolicy, EplbStats, TemporalEplbStats
from ._process import receive_planner_response, send_planner_request
from .planner import StairPlan, StairPlanner

_PLANNER_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


class StairEplbPolicy(AbstractEplbPolicy):
    """Plan raw logical load windows without running CPU search on the engine."""

    requires_temporal_load = True

    def __init__(self, config: StairConfig | None = None) -> None:
        self.config = config or StairConfig()
        self._process: subprocess.Popen[bytes] | None = None
        self._connection: Connection | None = None
        self._stderr: BinaryIO | None = None
        self._closed = False

    def _start_process(self) -> None:
        if self._closed:
            raise RuntimeError("STAIR planner is closed")
        if self._process is not None:
            if self._process.poll() is not None:
                details = self._process_error()
                self.close()
                raise RuntimeError(f"STAIR planner exited unexpectedly: {details}")
            return
        parent_socket, child_socket = socket.socketpair()
        environment = os.environ.copy()
        environment.update(_PLANNER_THREAD_ENV)
        environment["VLLM_PLUGINS"] = ""
        environment["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        stderr = tempfile.TemporaryFile()  # noqa: SIM115 - owned until close()
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "vllm.distributed.eplb.policy.stair._process",
                    str(child_socket.fileno()),
                ],
                pass_fds=(child_socket.fileno(),),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=stderr,
                env=environment,
            )
        except Exception:
            parent_socket.close()
            stderr.close()
            raise
        finally:
            child_socket.close()
        self._process = process
        self._connection = Connection(parent_socket.detach())
        self._stderr = stderr

    def _process_error(self) -> str:
        if self._stderr is None:
            return "no subprocess diagnostic"
        self._stderr.seek(0)
        return self._stderr.read(16384).decode(errors="replace").strip()

    def close(self) -> None:
        """Close the planner after the EPLB transfer worker has stopped."""
        self._closed = True
        connection, self._connection = self._connection, None
        process, self._process = self._process, None
        if connection is not None:
            connection.close()
        try:
            if process is not None:
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        finally:
            if self._stderr is not None:
                self._stderr.close()
                self._stderr = None

    def _plan_in_subprocess(
        self,
        loads: np.ndarray,
        current: np.ndarray,
        history: np.ndarray,
        node_ids: np.ndarray,
    ) -> StairPlan:
        self._start_process()
        assert self._connection is not None
        try:
            send_planner_request(
                self._connection,
                (loads, current, history, node_ids, asdict(self.config)),
            )
            error_type, error, fields = receive_planner_response(self._connection)
        except Exception as error:
            details = self._process_error()
            self.close()
            raise RuntimeError(
                f"STAIR planner subprocess communication failed: {details}"
            ) from error
        if error is not None:
            error_class = ValueError if error_type == "ValueError" else RuntimeError
            raise error_class(f"STAIR planner subprocess failed:\n{error.rstrip()}")
        if fields is None:
            raise RuntimeError("STAIR planner returned no placement")
        return StairPlan(*fields)

    @staticmethod
    def _gather_layer_shard(
        local_values: torch.Tensor, num_layers: int, group: ProcessGroup
    ) -> torch.Tensor:
        group_size = group.size()
        if group_size == 1:
            return local_values.clone()
        capacity = (num_layers + group_size - 1) // group_size
        padded = local_values.new_zeros((capacity, *local_values.shape[1:]))
        padded[: local_values.shape[0]].copy_(local_values)
        shards = [torch.empty_like(padded) for _ in range(group_size)]
        torch.distributed.all_gather(shards, padded, group=group)
        merged = local_values.new_empty((num_layers, *local_values.shape[1:]))
        for owner, shard in enumerate(shards):
            merged[owner::group_size].copy_(shard[: len(merged[owner::group_size])])
        return merged

    def plan_sharded_rebalance(
        self,
        loads: np.ndarray,
        current: np.ndarray,
        history: np.ndarray,
        node_ids: np.ndarray,
        group: ProcessGroup,
    ) -> StairPlan:
        """Plan disjoint layer shards and collectively propagate local failures."""
        local_error: Exception | None = None
        local_fields: tuple[torch.Tensor, torch.Tensor] | None = None
        num_layers = 0
        try:
            if current.ndim != 3 or not current.shape[0]:
                raise ValueError(
                    "STAIR placement must be nonempty [layers, ranks, slots]"
                )
            num_layers = current.shape[0]
            owned = np.arange(group.rank(), num_layers, group.size())
            if owned.size:
                local = self._plan_in_subprocess(
                    loads[:, owned], current[owned], history[owned], node_ids
                )
                if local.imbalance_ratios is None:
                    raise RuntimeError("STAIR planner returned no imbalance summary")
                local_fields = (
                    torch.from_numpy(
                        np.stack(
                            (
                                local.rank_expert_ids,
                                local.source_rank_ids,
                                local.source_slot_ids,
                            ),
                            axis=1,
                        )
                    ),
                    torch.from_numpy(
                        np.column_stack(
                            (local.predicted_mean_ratios, local.imbalance_ratios)
                        )
                    ),
                )
            else:
                local_fields = (
                    torch.empty((0, 3, *current.shape[1:]), dtype=torch.int64),
                    torch.empty((0, 5), dtype=torch.float64),
                )
        except Exception as error:
            local_error = error
        succeeded = torch.tensor(local_error is None, dtype=torch.int32)
        torch.distributed.all_reduce(
            succeeded, op=torch.distributed.ReduceOp.MIN, group=group
        )
        if not succeeded.item():
            if local_error is not None:
                raise local_error
            raise RuntimeError("STAIR layer planning failed on another EPLB rank")
        assert local_fields is not None
        integers = self._gather_layer_shard(local_fields[0], num_layers, group).numpy()
        floats = self._gather_layer_shard(local_fields[1], num_layers, group).numpy()
        plan = StairPlan(
            rank_expert_ids=integers[:, 0],
            source_rank_ids=integers[:, 1],
            source_slot_ids=integers[:, 2],
            predicted_mean_ratios=floats[:, 0],
            imbalance_ratios=floats[:, 1:],
        )
        StairPlanner.validate_plan(
            current,
            plan,
            loads.shape[2],
            node_ids,
            self.config.rank_transfer_limit,
            self.config.cross_node_transfer_limit,
        )
        return plan

    def rebalance_experts(
        self, stats: EplbStats, old_global_expert_indices: torch.Tensor | None = None
    ) -> torch.Tensor | ExpertTransferPlan:
        if not isinstance(stats, TemporalEplbStats):
            raise ValueError("STAIR requires raw temporal load statistics")
        loads = stats.global_expert_load_window
        current_map = old_global_expert_indices
        if (
            loads.device.type != "cpu"
            or loads.ndim != 3
            or stats.sample_counts != (1,) * loads.shape[0]
            or current_map is None
            or current_map.device.type != "cpu"
            or current_map.shape != (loads.shape[1], stats.num_replicas)
            or stats.num_gpus < 1
            or stats.num_replicas % stats.num_gpus
        ):
            raise ValueError(
                "STAIR requires raw CPU temporal loads and a fixed-capacity placement"
            )
        group = get_eplb_group().cpu_group
        if group.size() != stats.num_gpus:
            raise ValueError(
                "STAIR planning group does not match the placement topology"
            )
        current = current_map.numpy().reshape(
            loads.shape[1], stats.num_gpus, stats.num_replicas // stats.num_gpus
        )
        plan = self.plan_sharded_rebalance(
            loads.numpy(),
            current,
            stats.last_committed_mean_ratios,
            np.asarray(stats.rank_node_ids, dtype=np.int64),
            group,
        )
        return ExpertTransferPlan(
            physical_to_logical_map=torch.from_numpy(
                plan.rank_expert_ids.reshape(current_map.shape)
            ).to(dtype=current_map.dtype),
            source_rank_ids=plan.source_rank_ids,
            source_slot_ids=plan.source_slot_ids,
            predicted_mean_ratios=plan.predicted_mean_ratios,
        )
