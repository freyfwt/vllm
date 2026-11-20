# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import math
from statistics import NormalDist
from typing import Any

from pydantic import ConfigDict, field_validator, model_validator

from vllm.config.utils import config


@config(config=ConfigDict(frozen=True))
class StairConfig:
    """Tuning for the CPU STAIR placement planner.

    Covariance-aware risk and hysteresis are mandatory policy behavior. Balance
    is the reciprocal of the mean max-to-average rank-load ratio.

    """

    load_window_bins: int = 64
    """Maximum chronological bins for searching the load window. All raw steps
    remain available for acceptance and hysteresis checks."""

    load_risk_quantile: float = 0.75
    """One-sided standard-normal quantile for mean-plus-deviation expert and
    rank risk."""

    relative_balance_threshold: float = 0.95
    """Rebalance when current balance divided by last committed balance is
    at or below this value."""

    absolute_balance_threshold: float = 0.90
    """Rebalance when current balance is at or below this value."""

    rank_transfer_limit: int = -1
    """Maximum incoming and outgoing expert transfers per rank and layer.
    Minus one removes this limit."""

    cross_node_transfer_limit: int = -1
    """Maximum incoming and outgoing cross-node transfers per node and layer.
    Minus one removes this limit; zero disables cross-node transfers."""

    replica_search_num_stages: int = 4
    """Number of risk-ordered expert groups in the replica search."""

    replica_search_radius: int = 8
    """Maximum distance from the greedy extra-replica budget explored at
    each search stage."""

    replica_search_beam_size: int = 64
    """Maximum unique replica-count candidates retained after each stage."""

    placement_search_backtrack_limit: int = 32
    """Maximum feasible-branch reversals in constrained LPT placement.
    Zero disables backtracking."""

    @property
    def z_score(self) -> float:
        return NormalDist().inv_cdf(self.load_risk_quantile)

    @field_validator("*", mode="before")
    @classmethod
    def _reject_bool(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("STAIR numeric fields must not be booleans")
        return value

    @model_validator(mode="after")
    def _validate(self):
        if not 2 <= self.load_window_bins <= 256:
            raise ValueError("stair_config.load_window_bins must be between 2 and 256")
        if (
            not math.isfinite(self.load_risk_quantile)
            or not 0.5 < self.load_risk_quantile < 1
        ):
            raise ValueError(
                "stair_config.load_risk_quantile must be between 0.5 and one"
            )
        for name in ("relative_balance_threshold", "absolute_balance_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"stair_config.{name} must be between zero and one")
        if self.rank_transfer_limit != -1 and self.rank_transfer_limit < 1:
            raise ValueError("stair_config.rank_transfer_limit must be -1 or positive")
        if self.cross_node_transfer_limit < -1:
            raise ValueError(
                "stair_config.cross_node_transfer_limit must be at least -1"
            )
        if not 1 <= self.replica_search_num_stages <= 8:
            raise ValueError(
                "stair_config.replica_search_num_stages must be between 1 and 8"
            )
        if not 0 <= self.replica_search_radius <= 32:
            raise ValueError(
                "stair_config.replica_search_radius must be between 0 and 32"
            )
        if not 1 <= self.replica_search_beam_size <= 128:
            raise ValueError(
                "stair_config.replica_search_beam_size must be between 1 and 128"
            )
        if not 0 <= self.placement_search_backtrack_limit <= 64:
            raise ValueError(
                "stair_config.placement_search_backtrack_limit must be between 0 and 64"
            )
        return self
