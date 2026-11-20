# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""STAIR planning from explicit CPU snapshots, independent of worker state."""

import numpy as np
import torch

from vllm.config.eplb import StairConfig
from vllm.distributed.eplb.transfer_plan import ExpertTransferPlan

from ..abstract import AbstractEplbPolicy, EplbStats, TemporalEplbStats
from .planner import StairPlanner


class StairEplbPolicy(AbstractEplbPolicy):
    """Compute a local layer shard without process groups or child processes."""

    requires_temporal_load = True
    shard_by_layer = True

    def __init__(self, config: StairConfig | None = None) -> None:
        self.config = config or StairConfig()

    def rebalance_experts(
        self, stats: EplbStats, old_global_expert_indices: torch.Tensor | None = None
    ) -> ExpertTransferPlan:
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
        current = current_map.numpy().reshape(
            loads.shape[1], stats.num_gpus, stats.num_replicas // stats.num_gpus
        )
        node_ids = np.asarray(stats.rank_node_ids, dtype=np.int64)
        plan = StairPlanner.plan_rebalance(
            loads.numpy(),
            current,
            stats.last_committed_mean_ratios,
            node_ids,
            self.config,
        )
        StairPlanner.validate_plan(
            current,
            plan,
            loads.shape[2],
            node_ids,
            self.config.rank_transfer_limit,
            self.config.cross_node_transfer_limit,
        )
        return ExpertTransferPlan(
            physical_to_logical_map=torch.from_numpy(
                plan.rank_expert_ids.reshape(current_map.shape)
            ).to(dtype=current_map.dtype),
            source_rank_ids=plan.source_rank_ids,
            source_slot_ids=plan.source_slot_ids,
            predicted_mean_ratios=plan.predicted_mean_ratios,
        )
