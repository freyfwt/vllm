# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterator

import torch

from vllm.triton_utils import tl, triton

GUMBEL_BLOCK_SIZE = 1024
MAX_TRITON_PROGRAMS_PER_LAUNCH = 65_535
_FP32_UNIT_ROUNDOFF: tl.constexpr = 2.0**-24
_FP64_UNIT_ROUNDOFF: tl.constexpr = 2.0**-53
_LOG1P_NEG_SERIES_CUTOFF: tl.constexpr = 0.25


@triton.jit
def _temperature_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    temperature_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    if temperature == 0.0 or temperature == 1.0:
        # Early return to avoid loading logits.
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size

    logits = tl.load(logits_ptr + token_idx * logits_stride + block, mask=mask)
    logits = logits.to(tl.float32)
    logits = logits / temperature
    tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


def apply_temperature(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
) -> None:
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 8192
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    _temperature_kernel[(num_tokens, num_blocks)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


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
def murmur3_hash32(seed, pos, offset, domain: tl.constexpr = 0):
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
def murmur3_uniform32(seed, pos, offset):
    random24 = murmur3_hash32(seed, pos, offset) >> 8
    # random24 fits in signed int32. The intermediate cast also avoids a
    # uint32-to-float conversion that is unsupported by Ascend BiShengIR.
    return (random24.to(tl.int32).to(tl.float32) + 0.5) * _FP32_UNIT_ROUNDOFF


@triton.jit
def murmur3_uniform64(seed, pos, offset):
    lo = murmur3_hash32(seed, pos, offset).to(tl.uint64)
    hi = murmur3_hash32(seed, pos, offset, domain=0x9E3779B9).to(tl.uint64)
    random53 = ((hi << 32) | lo) >> 11
    return (random53.to(tl.float64) + 0.5) * _FP64_UNIT_ROUNDOFF


@triton.jit
def _log1p_neg(value):
    """Compute log1p(-value) without relying on a backend libdevice."""
    # Horner form of -sum(value**k / k, k=1..8). Its absolute error stays
    # below 6e-7 on [0, 0.25]; direct subtraction is well-conditioned above it.
    polynomial = 1.0 / 8.0
    polynomial = 1.0 / 7.0 + value * polynomial
    polynomial = 1.0 / 6.0 + value * polynomial
    polynomial = 1.0 / 5.0 + value * polynomial
    polynomial = 1.0 / 4.0 + value * polynomial
    polynomial = 1.0 / 3.0 + value * polynomial
    polynomial = 1.0 / 2.0 + value * polynomial
    polynomial = 1.0 + value * polynomial
    series = -value * polynomial

    direct = tl.log(tl.maximum(1.0 - value, _FP32_UNIT_ROUNDOFF))
    return tl.where(value < _LOG1P_NEG_SERIES_CUTOFF, series, direct)


@triton.jit
def gumbel_block_argmax(
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
        # Apply temperature.
        # NOTE(woosuk): Match the behavior of _temperature_kernel.
        # E.g., if the kernel uses tl.div_rn, we should use tl.div_rn here too.
        logits = logits / temp

    if processed_logits_ptr is not None:
        # Store the temperature-applied logits.
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

    # fp32 is the default reduction dtype; fp64 is ~1/32–1/64x the throughput
    # on H100/Ada/Blackwell and empirically indistinguishable for Gumbel-max.
    if USE_FP64:
        logits = logits.to(tl.float64)
    if temp != 0.0:
        seed = tl.load(seeds_ptr + req_state_idx, mask=is_valid_req, other=0)
        pos = tl.load(pos_ptr + token_idx)

        if USE_FP64:
            u = murmur3_uniform64(seed, pos, block)
            gumbel_noise = -tl.log(-tl.log(u))
        else:
            u = murmur3_uniform32(seed, pos, block)
            # Reflect the winning tail from u -> 1 to the denser fp32 region
            # near u -> 0. _log1p_neg avoids cancellation after reflection.
            gumbel_noise = -tl.log(-_log1p_neg(u))

        # Apply gumbel noise.
        logits = tl.where(mask, logits + gumbel_noise, float("-inf"))

    value, idx = tl.max(logits, axis=0, return_indices=True)
    return value, idx


@triton.jit
def _gumbel_sample_kernel(
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
    )
    logits = logits.to(tl.float32)

    value, idx = gumbel_block_argmax(
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


def _token_launch_ranges(
    num_tokens: int,
    num_blocks: int,
) -> Iterator[tuple[int, int]]:
    tokens_per_launch = max(1, MAX_TRITON_PROGRAMS_PER_LAUNCH // num_blocks)
    for token_start in range(0, num_tokens, tokens_per_launch):
        yield token_start, min(token_start + tokens_per_launch, num_tokens)


def _reduce_block_argmax(
    local_argmax: torch.Tensor,
    local_max: torch.Tensor,
) -> torch.Tensor:
    # NOTE(woosuk): Use int64 for later indexing.
    max_block_idx = local_max.argmax(dim=-1, keepdim=True)
    return local_argmax.gather(dim=-1, index=max_block_idx).view(-1)


def gumbel_sample(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    expanded_idx_mapping: torch.Tensor,  # [num_tokens]
    temperature: torch.Tensor,  # [max_num_reqs]
    seed: torch.Tensor,  # [max_num_reqs]
    pos: torch.Tensor,  # [num_tokens]
    apply_temperature: bool,
    output_processed_logits: torch.Tensor | None = None,
    output_processed_logits_col: torch.Tensor | None = None,
    use_fp64: bool = False,
) -> torch.Tensor:
    # Enforce contiguity on non-strided input tensors
    expanded_idx_mapping = expanded_idx_mapping.contiguous()
    pos = pos.contiguous()
    if output_processed_logits_col is not None:
        output_processed_logits_col = output_processed_logits_col.contiguous()
    num_tokens, vocab_size = logits.shape
    num_blocks = triton.cdiv(vocab_size, GUMBEL_BLOCK_SIZE)
    local_argmax = logits.new_empty(num_tokens, num_blocks, dtype=torch.int64)
    local_max_dtype = torch.float64 if use_fp64 else torch.float32
    local_max = logits.new_empty(num_tokens, num_blocks, dtype=local_max_dtype)
    per_token_col = (
        output_processed_logits_col is not None
        and output_processed_logits_col.dim() > 0
    )
    processed_logits_stride = 0
    if output_processed_logits is not None:
        processed_logits_stride = output_processed_logits.stride(0)
    for token_start, token_end in _token_launch_ranges(num_tokens, num_blocks):
        token_slice = slice(token_start, token_end)
        processed_logits_col = output_processed_logits_col
        if per_token_col:
            assert output_processed_logits_col is not None
            processed_logits_col = output_processed_logits_col[token_slice]
        _gumbel_sample_kernel[(token_end - token_start, num_blocks)](
            local_argmax[token_slice],
            local_argmax.stride(0),
            local_max[token_slice],
            local_max.stride(0),
            output_processed_logits,
            processed_logits_stride,
            processed_logits_col,
            logits[token_slice],
            logits.stride(0),
            expanded_idx_mapping[token_slice],
            seed,
            pos[token_slice],
            temperature,
            vocab_size,
            BLOCK_SIZE=GUMBEL_BLOCK_SIZE,
            APPLY_TEMPERATURE=apply_temperature,
            USE_FP64=use_fp64,
            PER_TOKEN_COL=per_token_col,
        )
    return _reduce_block_argmax(local_argmax, local_max)
