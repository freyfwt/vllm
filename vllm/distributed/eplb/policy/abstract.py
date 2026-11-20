# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch

from vllm.distributed.eplb.transfer_plan import ExpertTransferPlan


@dataclass(slots=True)
class EplbStats:
    """Load and topology snapshot for one EPLB planning call.

    Policies may accept subclasses with additional planning inputs."""

    global_expert_load_window: torch.Tensor
    """
    Expert load window, converted to CPU before policy invocation.
    Built-in policies use shape (num_moe_layers, num_logical_experts).
    """
    num_replicas: int
    """
    Number of physical experts.
    """
    num_groups: int
    """
    Number of expert groups.
    """
    num_nodes: int
    """
    Number of nodes.
    """
    num_gpus: int
    """
    Number of GPUs.
    """


@dataclass(slots=True)
class TemporalEplbStats(EplbStats):
    """Chronological `[steps, layers, experts]` loads on a shared rank clock."""

    sample_counts: tuple[int, ...]
    """Number of original iterations represented by each sample."""
    rank_node_ids: tuple[int, ...]
    """Node ordinals in the same rank order as the placement."""
    last_committed_mean_ratios: np.ndarray
    """Per-layer committed predictions, copied before planning starts."""


class AbstractEplbPolicy(ABC):
    requires_temporal_load: bool = False
    shard_by_layer: bool = False
    """Let the EPLB runtime partition temporal inputs and gather transfer plans."""

    def close(self) -> None:
        """Idempotently release resources after calls into the policy have stopped."""
        return None

    @abstractmethod
    def rebalance_experts(
        self,
        stats: EplbStats,
        old_global_expert_indices: torch.Tensor | None = None,
    ) -> torch.Tensor | ExpertTransferPlan:
        """Entry point for expert-parallelism load balancer.

        Args:
            stats: Load/topology snapshot with a CPU load tensor.
                Built-in policies use `[layers, num_logical_experts]` loads.
                Policy-specific subclasses may provide additional inputs.
            old_global_expert_indices: [layers, num_replicas], the old global
                expert indices. Used to avoid unnecessary weight copying
                for experts moving within one rank.

        Returns:
            physical_to_logical_map: [layers, num_replicas], the expert
                index of each replica, on CPU. Policies requiring exact copy
                sources may return an `ExpertTransferPlan` instead.

        """
        raise NotImplementedError
