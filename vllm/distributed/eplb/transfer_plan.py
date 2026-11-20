# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""A policy placement with exact expert-copy sources."""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class ExpertTransferPlan:
    """Dense, fixed-capacity placement and sources in the previous placement.

    The CPU target map is `[layers, physical_experts]`; source arrays are
    `[layers, ranks, slots]`. Each rank contains at most one copy of an expert.
    Experts already present on a destination rank keep their original slot and
    use that local source. These restrictions let all communicator backends use
    their existing expert-to-row lookup. Elastic rank remapping is unsupported.

    Optional predicted mean max-to-average load ratios are `[layers]`. Changed
    layers have finite ratios; unchanged layers have NaN. They become history
    only after the corresponding weights and routing maps are committed.
    """

    physical_to_logical_map: torch.Tensor
    source_rank_ids: np.ndarray
    source_slot_ids: np.ndarray
    predicted_mean_ratios: np.ndarray | None = None

    def validate(
        self, old_physical_to_logical_map: torch.Tensor, layer_idx: int | None = None
    ) -> None:
        """Validate the full plan, or one layer before its transfer starts."""
        target = self.physical_to_logical_map
        if (
            target.device.type != "cpu"
            or old_physical_to_logical_map.device.type != "cpu"
            or target.ndim != 2
            or 0 in target.shape
            or target.dtype not in (torch.int32, torch.int64)
            or old_physical_to_logical_map.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Explicit EPLB maps must be non-empty CPU integer tensors")
        ranks, slots = self.source_rank_ids, self.source_slot_ids
        if (
            not isinstance(ranks, np.ndarray)
            or not isinstance(slots, np.ndarray)
            or ranks.ndim != 3
            or ranks.shape != slots.shape
            or 0 in ranks.shape
            or ranks.shape[0] != target.shape[0]
            or ranks.shape[1] * ranks.shape[2] != target.shape[1]
            or not np.issubdtype(ranks.dtype, np.integer)
            or not np.issubdtype(slots.dtype, np.integer)
        ):
            raise ValueError(
                "Explicit EPLB sources must be integer [layers, ranks, slots] arrays"
            )
        num_layers, num_ranks, num_slots = ranks.shape
        if layer_idx is None:
            if old_physical_to_logical_map.shape != target.shape:
                raise ValueError("Explicit EPLB maps must have matching fixed capacity")
            selected = range(num_layers)
            old = old_physical_to_logical_map.numpy().reshape(ranks.shape)
        else:
            if (
                not 0 <= layer_idx < num_layers
                or old_physical_to_logical_map.shape != target.shape[1:]
            ):
                raise ValueError("Explicit EPLB layer does not match the plan")
            selected = range(layer_idx, layer_idx + 1)
            old = old_physical_to_logical_map.numpy().reshape(1, num_ranks, num_slots)
        target_values = target.numpy().reshape(ranks.shape)
        ratios = self.predicted_mean_ratios
        if ratios is not None and (
            not isinstance(ratios, np.ndarray)
            or ratios.shape != (num_layers,)
            or not np.issubdtype(ratios.dtype, np.floating)
        ):
            raise ValueError(
                "Predicted EPLB ratios must be one floating-point value per layer"
            )
        for old_layer, layer in zip(old, selected):
            target_layer = target_values[layer]
            source_ranks, source_slots = ranks[layer], slots[layer]
            if (
                np.any(old_layer < 0)
                or np.any(target_layer < 0)
                or np.any(source_ranks < 0)
                or np.any(source_ranks >= num_ranks)
                or np.any(source_slots < 0)
                or np.any(source_slots >= num_slots)
            ):
                raise ValueError(
                    "Explicit EPLB plans require dense maps and in-range sources"
                )
            for rank in range(num_ranks):
                if (
                    len(np.unique(old_layer[rank])) != num_slots
                    or len(np.unique(target_layer[rank])) != num_slots
                ):
                    raise ValueError(
                        "Explicit EPLB plans forbid rank-local duplicate experts"
                    )
                old_slots = {
                    expert: slot for slot, expert in enumerate(old_layer[rank])
                }
                for slot, expert in enumerate(target_layer[rank]):
                    previous_slot = old_slots.get(expert)
                    if previous_slot is not None and (
                        slot != previous_slot
                        or source_ranks[rank, slot] != rank
                        or source_slots[rank, slot] != previous_slot
                    ):
                        raise ValueError(
                            "Explicit EPLB plans must retain local experts in place"
                        )
            if not np.array_equal(old_layer[source_ranks, source_slots], target_layer):
                raise ValueError("Explicit EPLB source does not own its target expert")
            if not np.array_equal(np.unique(old_layer), np.unique(target_layer)):
                raise ValueError(
                    "Explicit EPLB plans must preserve logical-expert coverage"
                )
            if ratios is not None:
                changed = not np.array_equal(old_layer, target_layer)
                ratio = ratios[layer]
                if (changed and (not np.isfinite(ratio) or ratio < 1)) or (
                    not changed and not np.isnan(ratio)
                ):
                    raise ValueError(
                        "Predicted ratios require finite for changed, NaN otherwise"
                    )
