# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare legacy STAIR IPC with the shared policy process on real Gloo.

Run from the checkout being measured so child interpreters import that runtime.
The legacy mode targets the original STAIR fork; shared/direct target this tree.
"""

import argparse
import hashlib
import json
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.multiprocessing as mp

from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.config.parallel import EPLBConfig
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.distributed.eplb.policy import TemporalEplbStats
from vllm.distributed.eplb.policy.stair import StairConfig, StairPlanner
from vllm.distributed.parallel_state import get_eplb_group


def worker(rank, args):
    torch.set_num_threads(1)
    parallel = ParallelConfig(
        tensor_parallel_size=args.ranks,
        distributed_executor_backend="mp",
        eplb_config=EPLBConfig(policy="stair", use_async=args.transport != "direct"),
    )
    config = VllmConfig(parallel_config=parallel)
    parallel.enable_eplb = True
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=args.ranks,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"file://{args.output}/store",
            backend="gloo",
        )
        initialize_model_parallel(tensor_model_parallel_size=args.ranks, backend="gloo")
        state = EplbState(parallel, torch.device("cpu"))
        try:
            if args.transport == "legacy" and not hasattr(
                state.policy, "plan_sharded_rebalance"
            ):
                raise ValueError("Legacy mode requires the pre-integration STAIR tree")
            rng = np.random.default_rng(77)
            means = (
                np.geomspace(4, 80, args.experts)
                if args.skewed
                else np.full(args.experts, 20)
            )
            loads = rng.poisson(means, (args.steps, args.layers, args.experts))
            current = np.tile(np.arange(args.replicas) % args.experts, (args.layers, 1))
            nodes = np.arange(args.ranks) // max(1, args.ranks // 2)
            history = np.full(args.layers, np.nan)
            stats = TemporalEplbStats(
                torch.from_numpy(loads),
                args.replicas,
                1,
                len(set(nodes)),
                args.ranks,
                (1,) * args.steps,
                tuple(nodes),
                history,
            )
            old = torch.from_numpy(current)
            expected = StairPlanner.plan_rebalance(
                loads,
                current.reshape(args.layers, args.ranks, -1),
                history,
                nodes,
                StairConfig(),
            )
            startup = time.perf_counter()
            if args.transport == "shared":
                from vllm.distributed.eplb.policy_process import EplbPolicyProcess

                state.policy_process = EplbPolicyProcess(state.policy)
            startup_ms = (time.perf_counter() - startup) * 1000
            plan = (
                state.policy.rebalance_experts
                if args.transport == "legacy"
                else state.rebalance_experts
            )
            cold = time.perf_counter()
            actual = plan(stats, old)
            cold_ms = (time.perf_counter() - cold) * 1000 + startup_ms
            for name, ref in (
                (
                    "physical_to_logical_map",
                    expected.rank_expert_ids.reshape(current.shape),
                ),
                ("source_rank_ids", expected.source_rank_ids),
                ("source_slot_ids", expected.source_slot_ids),
                ("predicted_mean_ratios", expected.predicted_mean_ratios),
            ):
                np.testing.assert_array_equal(getattr(actual, name), ref)
            group = get_eplb_group().cpu_group
            durations = []
            foreground = []
            scheduling = []
            descendants = psutil.Process().children(recursive=True)
            initial_rss = sum(p.memory_info().rss for p in descendants)
            initial_pageins = sum(
                getattr(p.memory_info(), "pageins", 0) for p in descendants
            )
            initial_cpu = sum(
                p.cpu_times().user + p.cpu_times().system for p in descendants
            )

            def measure_plans():
                nonlocal actual
                for _ in range(args.iterations):
                    torch.distributed.barrier(group=group)
                    started = time.perf_counter()
                    actual = plan(stats, old)
                    elapsed = torch.tensor((time.perf_counter() - started) * 1000)
                    torch.distributed.all_reduce(
                        elapsed, op=torch.distributed.ReduceOp.MAX, group=group
                    )
                    durations.append(elapsed.item())

            if args.foreground:
                done = threading.Event()
                errors = []

                def background():
                    try:
                        measure_plans()
                    except Exception as error:
                        errors.append(error)
                    finally:
                        done.set()

                thread = threading.Thread(target=background)
                tensor = torch.zeros(1)
                deadline = time.perf_counter()
                thread.start()
                while not done.wait(max(0, deadline - time.perf_counter())):
                    before = time.perf_counter()
                    scheduling.append(max(0, before - deadline) * 1e6)
                    sum(i * i for i in range(256))
                    tensor.add_(1)
                    foreground.append((time.perf_counter() - before) * 1e6)
                    deadline = max(deadline + 0.001, time.perf_counter())
                thread.join()
                if errors:
                    raise errors[0]
            else:
                measure_plans()
            descendants = psutil.Process().children(recursive=True)
            rss = sum(p.memory_info().rss for p in descendants)
            digest = hashlib.sha256()
            for value in (
                actual.physical_to_logical_map,
                actual.source_rank_ids,
                actual.source_slot_ids,
                actual.predicted_mean_ratios,
            ):
                digest.update(np.asarray(value).tobytes())
            expected_digest = hashlib.sha256()
            for value in (
                expected.rank_expert_ids.reshape(current.shape),
                expected.source_rank_ids,
                expected.source_slot_ids,
                expected.predicted_mean_ratios,
            ):
                expected_digest.update(np.asarray(value).tobytes())
            assert digest.digest() == expected_digest.digest()
            result = dict(
                rank=rank,
                transport=args.transport,
                startup_ms=startup_ms,
                cold_ms=cold_ms,
                planning_ms=durations,
                median_ms=float(np.median(durations)),
                p95_ms=float(np.percentile(durations, 95)),
                child_count=len(descendants),
                child_rss_mib=rss / 1024**2,
                initial_child_rss_mib=initial_rss / 1024**2,
                child_pageins=sum(
                    getattr(p.memory_info(), "pageins", 0) for p in descendants
                )
                - initial_pageins,
                child_cpu_ms=(
                    sum(p.cpu_times().user + p.cpu_times().system for p in descendants)
                    - initial_cpu
                )
                * 1000,
                digest=digest.hexdigest(),
                changed_slots=int(
                    np.count_nonzero(
                        np.asarray(actual.physical_to_logical_map) != current
                    )
                ),
                exact_reference_match=True,
                foreground=(
                    dict(
                        count=len(foreground),
                        submission_p99_us=float(np.percentile(foreground, 99)),
                        scheduling_p99_us=float(np.percentile(scheduling, 99)),
                    )
                    if foreground
                    else None
                ),
            )
            (args.output / f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        finally:
            if getattr(state, "policy_process", None) is not None:
                state.policy_process.close()
            state.policy.close()
            cleanup_dist_env_and_memory()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transport", choices=("legacy", "shared", "direct"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--replicas", type=int, default=20)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--skewed", action="store_true")
    parser.add_argument("--foreground", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mp.spawn(worker, args=(args,), nprocs=args.ranks, join=True)
    results = [
        json.loads((args.output / f"rank-{rank}.json").read_text())
        for rank in range(args.ranks)
    ]
    assert len({r["digest"] for r in results}) == 1
    result = dict(
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        ranks=results,
    )
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
