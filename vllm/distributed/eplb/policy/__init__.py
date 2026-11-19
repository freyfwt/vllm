# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import get_args

from vllm.config.parallel import EPLBPolicyOption

from .abstract import AbstractEplbPolicy, EplbStats, TemporalEplbStats
from .default import DefaultEplbPolicy
from .stair import StairEplbPolicy

EPLB_POLICIES = {"default": DefaultEplbPolicy, "stair": StairEplbPolicy}

# Ensure that the EPLB_POLICIES keys match the EPLBPolicyOption values
assert set(EPLB_POLICIES.keys()) == set(get_args(EPLBPolicyOption))

__all__ = [
    "AbstractEplbPolicy",
    "EplbStats",
    "TemporalEplbStats",
    "DefaultEplbPolicy",
    "StairEplbPolicy",
    "EPLB_POLICIES",
]
