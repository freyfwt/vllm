#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the original Philox and optimized Murmur3 Gumbel samplers on GPU."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

BASELINE_COMMIT = "f5a8d73377d0f0a4e00cba172f9fbd0d50471b07"
DEFAULT_SEED = 20_260_803
DEFAULT_WARMUPS = 10
DEFAULT_ROUNDS = 100
BLOCK_SIZE = 1024
IMPLEMENTATIONS = ("philox", "murmur3")
SUITE_SHAPES = (
    (1, 20),
    (1, 151_936),
    (8, 151_936),
    (64, 151_936),
    (128, 163_840),
    (128, 262_144),
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "gumbel_murmur3_gpu.json"

_TL_RAND_MIN = tl.constexpr(4.6566127342e-10)


@triton.jit
def _philox_rand64(seed, offset, includes_zero: tl.constexpr):
    lo, hi, _, _ = tl.randint4x(seed, offset)
    lo = lo.to(tl.uint32, bitcast=True).to(tl.uint64)
    hi = hi.to(tl.uint32, bitcast=True).to(tl.uint64)
    random64 = (hi << 32) | lo
    uniform = random64.to(tl.float64) * 5.421010862427522e-20
    if not includes_zero:
        uniform = tl.maximum(uniform, 2.2250738585072014e-308)
    return uniform


@triton.jit
def _philox_rand32(seed, offset, includes_zero: tl.constexpr):
    uniform = tl.rand(seed, offset)
    if not includes_zero:
        uniform = tl.maximum(uniform, _TL_RAND_MIN)
    return uniform


@triton.jit
def _philox_block_argmax(
    logits,
    block,
    mask,
    token_idx,
    expanded_idx_mapping_ptr,
    temp_ptr,
    seeds_ptr,
    pos_ptr,
    processed_logits_ptr,
    processed_logits_stride,
    processed_logits_col_ptr,
    vocab_size,
    APPLY_TEMPERATURE: tl.constexpr,
    USE_FP64: tl.constexpr,
    PER_TOKEN_COL: tl.constexpr = False,
):
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx).to(tl.int64)
    is_valid_req = req_state_idx >= 0
    temp = tl.load(temp_ptr + req_state_idx, mask=is_valid_req, other=0.0).to(
        tl.float32
    )
    if temp != 0.0 and APPLY_TEMPERATURE:
        logits = logits / temp

    if processed_logits_ptr is not None:
        if processed_logits_col_ptr is not None:
            if PER_TOKEN_COL:
                col = tl.load(processed_logits_col_ptr + token_idx)
            else:
                col = tl.load(processed_logits_col_ptr)
        else:
            col = 0
        tl.store(
            processed_logits_ptr
            + req_state_idx * processed_logits_stride
            + col * vocab_size
            + block,
            logits,
            mask=mask & is_valid_req,
        )

    if USE_FP64:
        logits = logits.to(tl.float64)
    if temp != 0.0:
        seed = tl.load(seeds_ptr + req_state_idx, mask=is_valid_req, other=0)
        pos = tl.load(pos_ptr + token_idx)
        gumbel_seed = tl.randint(seed, pos)

        if USE_FP64:
            uniform = _philox_rand64(gumbel_seed, block, includes_zero=False)
            noise = -tl.log(-tl.log(uniform))
        else:
            uniform = _philox_rand32(gumbel_seed, block, includes_zero=False)
            noise = -tl.log(-tldevice.log1p(-uniform))
        logits = tl.where(mask, logits + noise, float("-inf"))

    return tl.max(logits, axis=0, return_indices=True)


@triton.jit
def _philox_sample_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    processed_logits_ptr,
    processed_logits_stride,
    processed_logits_col_ptr,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
    USE_FP64: tl.constexpr,
    PER_TOKEN_COL: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        logits_ptr + token_idx * logits_stride + block,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)

    value, idx = _philox_block_argmax(
        logits,
        block,
        mask,
        token_idx,
        expanded_idx_mapping_ptr,
        temp_ptr,
        seeds_ptr,
        pos_ptr,
        processed_logits_ptr,
        processed_logits_stride,
        processed_logits_col_ptr,
        vocab_size,
        APPLY_TEMPERATURE=APPLY_TEMPERATURE,
        USE_FP64=USE_FP64,
        PER_TOKEN_COL=PER_TOKEN_COL,
    )
    token_id = block_idx * BLOCK_SIZE + idx
    tl.store(local_argmax_ptr + token_idx * local_argmax_stride + block_idx, token_id)
    tl.store(local_max_ptr + token_idx * local_max_stride + block_idx, value)


def philox_sample(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
) -> torch.Tensor:
    """Run the original FP32 Philox path from BASELINE_COMMIT."""
    expanded_idx_mapping = expanded_idx_mapping.contiguous()
    pos = pos.contiguous()
    num_tokens, vocab_size = logits.shape
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    local_argmax = logits.new_empty(num_tokens, num_blocks, dtype=torch.int64)
    local_max = logits.new_empty(num_tokens, num_blocks, dtype=torch.float32)
    _philox_sample_kernel[(num_tokens, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        None,
        0,
        None,
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        seed,
        pos,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        APPLY_TEMPERATURE=False,
        USE_FP64=False,
        PER_TOKEN_COL=False,
    )
    max_block_idx = local_max.argmax(dim=-1, keepdim=True)
    return local_argmax.gather(dim=-1, index=max_block_idx).view(-1)


@triton.jit
def _murmur3_rotl32(value, shift: tl.constexpr):
    return (value << shift) | (value >> (32 - shift))


@triton.jit
def _murmur3_mix(h, key):
    key *= 0xCC9E2D51
    key = _murmur3_rotl32(key, 15)
    key *= 0x1B873593
    h ^= key
    h = _murmur3_rotl32(h, 13)
    return h * 5 + 0xE6546B64


@triton.jit
def _murmur3_fmix32(h):
    h ^= h >> 16
    h *= 0x85EBCA6B
    h ^= h >> 13
    h *= 0xC2B2AE35
    return h ^ (h >> 16)


@triton.jit
def _murmur3_hash32(seed, pos, offset, domain: tl.constexpr = 0):
    seed = seed.to(tl.int64)
    pos = pos.to(tl.int64)
    offset = offset.to(tl.uint32)
    h = offset ^ offset
    h ^= domain
    h = _murmur3_mix(h, (seed & 0xFFFFFFFF).to(tl.uint32))
    h = _murmur3_mix(h, ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32))
    h = _murmur3_mix(h, (pos & 0xFFFFFFFF).to(tl.uint32))
    h = _murmur3_mix(h, offset)
    return _murmur3_fmix32(h ^ 16)


@triton.jit
def _murmur3_uniform32(seed, pos, offset):
    random24 = _murmur3_hash32(seed, pos, offset) >> 8
    return (random24.to(tl.int32).to(tl.float32) + 0.5) * 5.960464477539063e-08


@triton.jit
def _murmur3_uniform64(seed, pos, offset):
    lo = _murmur3_hash32(seed, pos, offset).to(tl.uint64)
    hi = _murmur3_hash32(seed, pos, offset, domain=0x9E3779B9).to(tl.uint64)
    random53 = ((hi << 32) | lo) >> 11
    return (random53.to(tl.float64) + 0.5) * 1.1102230246251565e-16


@triton.jit
def _log1p_neg_stable(value):
    polynomial = 1.0 / 8.0
    polynomial = 1.0 / 7.0 + value * polynomial
    polynomial = 1.0 / 6.0 + value * polynomial
    polynomial = 1.0 / 5.0 + value * polynomial
    polynomial = 1.0 / 4.0 + value * polynomial
    polynomial = 1.0 / 3.0 + value * polynomial
    polynomial = 1.0 / 2.0 + value * polynomial
    polynomial = 1.0 + value * polynomial
    series = -value * polynomial

    direct = tl.log(tl.maximum(1.0 - value, 5.960464477539063e-08))
    return tl.where(value < 0.25, series, direct)


@triton.jit
def _murmur3_block_argmax(
    logits,
    block,
    mask,
    token_idx,
    expanded_idx_mapping_ptr,
    temp_ptr,
    seeds_ptr,
    pos_ptr,
    processed_logits_ptr,
    processed_logits_stride,
    processed_logits_col_ptr,
    vocab_size,
    APPLY_TEMPERATURE: tl.constexpr,
    USE_FP64: tl.constexpr,
    PER_TOKEN_COL: tl.constexpr = False,
):
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx).to(tl.int64)
    is_valid_req = req_state_idx >= 0
    temp = tl.load(temp_ptr + req_state_idx, mask=is_valid_req, other=0.0).to(
        tl.float32
    )
    if temp != 0.0 and APPLY_TEMPERATURE:
        logits = logits / temp

    if processed_logits_ptr is not None:
        if processed_logits_col_ptr is not None:
            if PER_TOKEN_COL:
                col = tl.load(processed_logits_col_ptr + token_idx)
            else:
                col = tl.load(processed_logits_col_ptr)
        else:
            col = 0
        tl.store(
            processed_logits_ptr
            + req_state_idx * processed_logits_stride
            + col * vocab_size
            + block,
            logits,
            mask=mask & is_valid_req,
        )

    if USE_FP64:
        logits = logits.to(tl.float64)
    if temp != 0.0:
        seed = tl.load(seeds_ptr + req_state_idx, mask=is_valid_req, other=0)
        pos = tl.load(pos_ptr + token_idx)

        if USE_FP64:
            uniform = _murmur3_uniform64(seed, pos, block)
            noise = -tl.log(-tl.log(uniform))
        else:
            uniform = _murmur3_uniform32(seed, pos, block)
            noise = -tl.log(-_log1p_neg_stable(uniform))
        logits = tl.where(mask, logits + noise, float("-inf"))

    return tl.max(logits, axis=0, return_indices=True)


@triton.jit
def _murmur3_sample_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    processed_logits_ptr,
    processed_logits_stride,
    processed_logits_col_ptr,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
    USE_FP64: tl.constexpr,
    PER_TOKEN_COL: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        logits_ptr + token_idx * logits_stride + block,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)

    value, idx = _murmur3_block_argmax(
        logits,
        block,
        mask,
        token_idx,
        expanded_idx_mapping_ptr,
        temp_ptr,
        seeds_ptr,
        pos_ptr,
        processed_logits_ptr,
        processed_logits_stride,
        processed_logits_col_ptr,
        vocab_size,
        APPLY_TEMPERATURE=APPLY_TEMPERATURE,
        USE_FP64=USE_FP64,
        PER_TOKEN_COL=PER_TOKEN_COL,
    )
    token_id = block_idx * BLOCK_SIZE + idx
    tl.store(local_argmax_ptr + token_idx * local_argmax_stride + block_idx, token_id)
    tl.store(local_max_ptr + token_idx * local_max_stride + block_idx, value)


def murmur3_sample(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
) -> torch.Tensor:
    expanded_idx_mapping = expanded_idx_mapping.contiguous()
    pos = pos.contiguous()
    num_tokens, vocab_size = logits.shape
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    local_argmax = logits.new_empty(num_tokens, num_blocks, dtype=torch.int64)
    local_max = logits.new_empty(num_tokens, num_blocks, dtype=torch.float32)
    _murmur3_sample_kernel[(num_tokens, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        None,
        0,
        None,
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        seed,
        pos,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        APPLY_TEMPERATURE=False,
        USE_FP64=False,
        PER_TOKEN_COL=False,
    )
    max_block_idx = local_max.argmax(dim=-1, keepdim=True)
    return local_argmax.gather(dim=-1, index=max_block_idx).view(-1)


Sampler = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]
SAMPLERS: dict[str, Sampler] = {
    "philox": philox_sample,
    "murmur3": murmur3_sample,
}


@dataclass
class TimingResult:
    implementation: str
    p50_ms: float
    mean_ms: float
    p90_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    raw_event_ms: list[float]


@dataclass
class CaseResult:
    batch_size: int
    vocab_size: int
    philox: TimingResult
    murmur3: TimingResult
    p50_speedup: float


@dataclass
class CapturedSampler:
    graph: Any
    output: torch.Tensor


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    upper_weight = position - lower
    return ordered[lower] * (1.0 - upper_weight) + ordered[upper] * upper_weight


def _make_inputs(
    batch_size: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    logits = torch.randn(
        (batch_size, vocab_size),
        dtype=torch.float32,
        device=device,
    )
    mapping = torch.arange(batch_size, dtype=torch.int32, device=device)
    temperature = torch.ones(batch_size, dtype=torch.float32, device=device)
    seed = torch.arange(batch_size, dtype=torch.int64, device=device) + DEFAULT_SEED
    pos = torch.arange(batch_size, dtype=torch.int64, device=device) + 1
    return logits, mapping, temperature, seed, pos


def _assert_output(
    output: torch.Tensor,
    batch_size: int,
    vocab_size: int,
    implementation: str,
) -> None:
    tokens = output.view(-1)
    if tuple(tokens.shape) != (batch_size,):
        raise AssertionError(f"{implementation}: unexpected output shape")
    in_range = (tokens >= 0) & (tokens < vocab_size)
    if not bool(in_range.all().cpu()):
        raise AssertionError(f"{implementation}: token is outside the vocabulary")


def _smoke_and_capture(
    name: str,
    inputs: tuple[torch.Tensor, ...],
    batch_size: int,
    vocab_size: int,
    device: torch.device,
) -> CapturedSampler:
    sampler = SAMPLERS[name]
    first = sampler(*inputs)
    torch.accelerator.synchronize()
    _assert_output(first, batch_size, vocab_size, name)

    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.accelerator.current_stream())
    with torch.cuda.stream(capture_stream):
        sampler(*inputs)
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        output = sampler(*inputs)
    torch.accelerator.current_stream().wait_stream(capture_stream)
    torch.accelerator.synchronize()

    graph.replay()
    torch.accelerator.synchronize()
    _assert_output(output, batch_size, vocab_size, name)
    return CapturedSampler(graph=graph, output=output)


def _rotated_order(index: int) -> tuple[str, ...]:
    offset = index % len(IMPLEMENTATIONS)
    return IMPLEMENTATIONS[offset:] + IMPLEMENTATIONS[:offset]


def _summarize(name: str, elapsed: list[float]) -> TimingResult:
    return TimingResult(
        implementation=name,
        p50_ms=statistics.median(elapsed),
        mean_ms=statistics.fmean(elapsed),
        p90_ms=_percentile(elapsed, 0.90),
        p99_ms=_percentile(elapsed, 0.99),
        min_ms=min(elapsed),
        max_ms=max(elapsed),
        raw_event_ms=elapsed,
    )


def _run_case(
    batch_size: int,
    vocab_size: int,
    device: torch.device,
    warmups: int,
    rounds: int,
) -> CaseResult:
    inputs = _make_inputs(batch_size, vocab_size, device)
    captured = {
        name: _smoke_and_capture(name, inputs, batch_size, vocab_size, device)
        for name in IMPLEMENTATIONS
    }

    for warmup in range(warmups):
        for name in _rotated_order(warmup):
            captured[name].graph.replay()
    torch.accelerator.synchronize()

    stream = torch.accelerator.current_stream()
    samples = {name: [] for name in IMPLEMENTATIONS}
    for round_index in range(rounds):
        for name in _rotated_order(round_index):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            captured[name].graph.replay()
            end.record(stream)
            end.synchronize()
            samples[name].append(start.elapsed_time(end))

    philox = _summarize("philox", samples["philox"])
    murmur3 = _summarize("murmur3", samples["murmur3"])
    return CaseResult(
        batch_size=batch_size,
        vocab_size=vocab_size,
        philox=philox,
        murmur3=murmur3,
        p50_speedup=philox.p50_ms / murmur3.p50_ms,
    )


def _git_output(repo_root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _write_results(
    path: Path,
    args: argparse.Namespace,
    device: torch.device,
    results: list[CaseResult],
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    properties = torch.cuda.get_device_properties(device)
    optimized_commit = _git_output(repo_root, "rev-parse", "HEAD")
    branch = _git_output(repo_root, "branch", "--show-current")
    dirty = bool(
        _git_output(repo_root, "status", "--porcelain", "--untracked-files=no")
    )
    payload = {
        "benchmark": "GPU Gumbel RNG performance comparison under CUDA Graph replay",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "device_index": args.device,
            "device_name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": properties.total_memory,
        },
        "software": {
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "triton_version": getattr(triton, "__version__", None),
        },
        "source": {
            "baseline": {
                "name": "original Philox FP32 Gumbel sampler",
                "commit": BASELINE_COMMIT,
            },
            "optimized": {
                "name": "current Murmur3 FP32 Gumbel sampler",
                "commit": optimized_commit,
                "branch": branch,
                "dirty_worktree": dirty,
            },
        },
        "configuration": {
            "dtype": "float32",
            "apply_temperature": False,
            "seed": DEFAULT_SEED,
            "block_size": BLOCK_SIZE,
            "input_boundary": "full-shape processed logits",
            "measurement": "one CUDA Graph replay",
            "p50_speedup_definition": "philox_p50_ms / murmur3_p50_ms",
            "warmups": args.warmups,
            "rounds": args.rounds,
            "implementations_interleaved": True,
            "shapes": [list(shape) for shape in SUITE_SHAPES],
        },
        "smoke_checks": [
            "output shape matches batch size",
            "sampled token IDs are inside the vocabulary",
        ],
        "results": [asdict(result) for result in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.device < 0:
        parser.error("--device must be non-negative")
    if args.warmups < 0 or args.rounds <= 0:
        parser.error("warmups must be non-negative and rounds must be positive")
    return args


def main() -> None:
    args = _parse_args()
    if not torch.accelerator.is_available():
        raise RuntimeError("A CUDA-capable NVIDIA GPU is required")
    if torch.accelerator.current_accelerator().type != "cuda":
        raise RuntimeError("A CUDA-capable NVIDIA GPU is required")
    if args.device >= torch.accelerator.device_count():
        raise ValueError(
            f"device {args.device} is unavailable; "
            f"found {torch.accelerator.device_count()} GPUs"
        )

    torch.accelerator.set_device_index(args.device)
    device = torch.device(f"cuda:{args.device}")
    torch.manual_seed(DEFAULT_SEED)
    print(
        f"device={torch.cuda.get_device_name(device)} "
        f"warmups={args.warmups} rounds={args.rounds}"
    )

    results = []
    for batch_size, vocab_size in SUITE_SHAPES:
        result = _run_case(batch_size, vocab_size, device, args.warmups, args.rounds)
        results.append(result)
        print(
            f"shape={batch_size}x{vocab_size} "
            f"philox={result.philox.p50_ms:.4f}ms "
            f"murmur3={result.murmur3.p50_ms:.4f}ms "
            f"speedup={result.p50_speedup:.2f}x"
        )
        gc.collect()
        torch.accelerator.empty_cache()

    _write_results(args.output_json, args, device, results)
    print(f"results={args.output_json.resolve()}")


if __name__ == "__main__":
    main()
