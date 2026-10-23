# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# v10: decoupled backward non-inner path.

import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.zeros import zero_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

_NUM_MPC = 60
_NON_INNER_TARGET_CTAS = _NUM_MPC * 8

_V10_DISABLED = os.environ.get("GEMS_SOFTMAX_BWD_V10", "0") == "1"

_DECOUPLED_MAX_TILE_ELEMS = 4096

_DECOUPLED_MIN_K = 256
_DECOUPLED_MIN_N = 256
_DECOUPLED_PATHOLOGIC_TILE_N = 16

# ---------------------------------------------------------------------------
# v11: single-pass inner backward for medium N (1024 < N < 16384)
#
# The upstream `softmax_backward_kernel_inner` autotunes `TILE_N` over
# {32,64,128,256,1024}, so its heuristic `ONE_TILE_PER_CTA = (TILE_N >= N)`
# is NEVER satisfied for N > 1024.  That forces the two-pass `else` branch,
# which traverses the `out` and `out_grad` arrays TWICE (once to reduce, once
# to apply) => 5 array traversals instead of the 3 an ideal single-pass needs.
#
# The v11 kernel keeps the whole row resident in registers, so it does
# 1 read of `out` + 1 read of `out_grad` + 1 write of `in_grad` = 3 traversals.
#
# It also uses a 1-D tile instead of the upstream 2-D [1, N] tile.  The 2-D
# shape lowers to `sizePerThread=[1,1]` => scalar 2-byte loads, while the 1-D
# tile lowers to `sizePerThread=[N/(32*num_warps)]` => vectorized loads.
# ---------------------------------------------------------------------------
_V11_DISABLED = os.environ.get("GEMS_SOFTMAX_BWD_V11", "0") == "1"

# Rows longer than this are handled by the two-pass large-N kernel, which
# streams the row in tiles instead of holding it in registers.
_V11_MAX_N = 8192

# Minimum N for the fused path.  For N <= 1024 the upstream inner kernel
# already hits ONE_TILE_PER_CTA=True and is single-pass, so leave it alone.
_V11_MIN_N = 1024

_V9_DISABLED = os.environ.get("GEMS_SOFTMAX_BWD_V8", "0") == "1"


def _num_mpc():
    try:
        return torch.musa.get_device_properties(
            torch.musa.current_device()
        ).multi_processor_count
    except Exception:
        return _NUM_MPC


def _upstream_backward_tile_n(M, N, K):
    num_sms = _num_mpc()
    tile_k = 1
    upper_bound = min(K, 8192)
    while tile_k <= upper_bound:
        num_blocks = M * triton.cdiv(K, tile_k)
        num_waves = num_blocks / num_sms
        if (num_waves > 1) and (tile_k * 2 <= upper_bound):
            tile_k *= 2
        else:
            break
    return tile_k, max(1, 1024 // tile_k)

def _next_pow2(n):
    n = int(max(1, n))
    return 1 << (n - 1).bit_length()

@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_non_inner"))
@triton.jit
def softmax_kernel_non_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_k = ext.program_id(1)
    pid_m = ext.program_id(0)

    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offset = pid_m * N * K + n_offsets[:, None] * K + k_offsets
        mask = (n_offsets[:, None] < N) & (k_offsets < K)
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        m = tl.max(inp, 0)
        e = tl.exp(inp - m[None, :])
        z = tl.sum(e, 0)
        out = (e / z).to(output_ptr.dtype.element_ty)
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out, mask=mask)
    else:
        m = tl.full([TILE_N, TILE_K], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N, TILE_K], value=0.0, dtype=tl.float32)

        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets
            mask = (n_offsets[:, None] < N) & (k_offsets < K)
            inp = tl.load(input_ptr + offsets, mask=mask, other=-float("-inf"))
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, 0)
        z = tl.sum(z * tl.exp(m - m_reduced[None, :]), 0)
        m = m_reduced

        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
            offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets
            mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
            inp = tl.load(input_ptr + offsets, mask=mask, other=-float("-inf"))
            o = tl.exp(inp - m[None, :]) / z[None, :]
            tl.store(output_ptr + offsets, o, mask=mask)

@triton.jit
def next_multiple_of(a, b):
    return tl.cdiv(a, b) * b

@triton.jit
def prev_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b

@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit
def softmax_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = ext.program_id(0)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offset = pid_m * N + n_offsets
        input_ptrs = input_ptr + offset
        mask = n_offsets < N
        inp = tl.load(input_ptrs, mask=mask, other=-float("-inf")).to(tl.float32)
        m = tl.max(inp, 0)
        e = tl.exp(inp - m)
        z = tl.sum(e, 0)
        out = (e / z).to(output_ptr.dtype.element_ty)
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out, mask=mask)
    else:
        m = tl.full([TILE_N], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N], value=0.0, dtype=tl.float32)
        input_ptr += pid_m * N
        output_ptr += pid_m * N

        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets)
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("-inf"))
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, 0)
        z = tl.sum(z * tl.exp(m - m_reduced), 0)
        m = m_reduced

        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, TILE_N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(
                input_ptr + n_offsets,
                mask=mask,
                other=-float("-inf"),
                eviction_policy="evict_first",
            )
            o = tl.exp(inp - m) / z
            tl.store(output_ptr + n_offsets, o, mask=mask)
        for start_n in range(TILE_N, N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets, eviction_policy="evict_first")
            o = tl.exp(inp - m) / z
            tl.store(output_ptr + n_offsets, o)

@libentry()
@triton.jit
def softmax_backward_1d_dot_kernel(
    out_ptr,
    out_grad_ptr,
    partial_sum_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    num_programs = tl.num_programs(0).to(tl.int64)

    acc = 0.0
    idx = pid * BLOCK_N
    stride = num_programs * BLOCK_N

    while idx < N:
        offsets = idx + tl.arange(0, BLOCK_N).to(tl.int64)
        mask = offsets < N

        y = tl.load(out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(out_grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        acc += tl.sum(y * dy, axis=0)
        idx += stride

    tl.store(partial_sum_ptr + pid, acc)

@libentry()
@triton.jit
def softmax_backward_1d_apply_kernel(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    scale,
    N,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    mask = offsets < N

    y = tl.load(out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(out_grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    dx = y * (dy - scale)
    tl.store(in_grad_ptr + offsets, dx.to(in_grad_ptr.dtype.element_ty), mask=mask)

def softmax_backward_1d_fast(grad_output, output, grad_input):
    N = output.numel()
    device = output.device

    BLOCK_N = 4096
    NUM_WARPS = 8
    NUM_BLOCKS_REDUCE = min(240, triton.cdiv(N, BLOCK_N))

    partial_sums = torch.empty(
        (NUM_BLOCKS_REDUCE,), dtype=torch.float32, device=device
    )
    softmax_backward_1d_dot_kernel[(NUM_BLOCKS_REDUCE, 1, 1)](
        output,
        grad_output,
        partial_sums,
        N,
        BLOCK_N=BLOCK_N,
        num_warps=NUM_WARPS,
    )

    scale_val = torch.sum(partial_sums).item()

    GRID_APPLY = triton.cdiv(N, BLOCK_N)
    softmax_backward_1d_apply_kernel[(GRID_APPLY, 1, 1)](
        output,
        grad_output,
        grad_input,
        scale_val,
        N,
        BLOCK_N=BLOCK_N,
        num_warps=NUM_WARPS,
    )
    return grad_input

@libentry()
@triton.jit
def softmax_backward_inner_packed_small_n_kernel(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
):
    pid_m = ext.program_id(0)
    base = pid_m.to(tl.int64) * (TILE_M * N)
    rows = tl.arange(0, TILE_M)
    n_offsets = tl.arange(0, TILE_N)

    offsets = base + rows[:, None] * N + n_offsets[None, :]
    m_mask = (pid_m * TILE_M + rows) < M
    mask = m_mask[:, None] & (n_offsets[None, :] < N)

    out_tile = tl.load(out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    scale = tl.sum(out_tile * out_grad_tile, axis=1)
    in_grad_tile = out_tile * (out_grad_tile - scale[:, None])
    tl.store(in_grad_ptr + offsets, in_grad_tile.to(in_grad_ptr.dtype.element_ty), mask=mask)

def _packed_small_n_tiles(N):
    tile_n = _next_pow2(N)
    if tile_n >= 16:
        return 32, tile_n
    return max(1, 1024 // tile_n), tile_n


# ---------------------------------------------------------------------------
# v11 single-pass inner backward kernel.
#
# One CTA owns one row of length N and keeps the whole row in registers:
#   load out (1x) + load out_grad (1x) -> reduce -> store in_grad (1x)
# = 3 array traversals, versus the 5 traversals of the upstream two-pass
# `else` branch, which re-reads both input arrays after the reduction.
#
# The row is held as a 1-D [TILE_N] fp32 vector, so TILE_N is capped at
# 8192 to keep the register footprint bounded (8192 fp32 = 32 KiB per CTA).
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def softmax_backward_inner_fused_kernel(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
):
    pid_m = ext.program_id(0)

    n_offsets = tl.arange(0, TILE_N)
    mask = n_offsets < N

    base = pid_m.to(tl.int64) * N

    out_tile = tl.load(out_ptr + base + n_offsets, mask=mask, other=0.0).to(tl.float32)
    out_grad_tile = tl.load(
        out_grad_ptr + base + n_offsets, mask=mask, other=0.0
    ).to(tl.float32)

    scale = tl.sum(out_tile * out_grad_tile, axis=0)
    in_grad_tile = out_tile * (out_grad_tile - scale)

    tl.store(
        in_grad_ptr + base + n_offsets,
        in_grad_tile.to(in_grad_ptr.dtype.element_ty),
        mask=mask,
    )


def _inner_fused_launch(tile_n):
    """Pick num_warps so that each thread owns ~16 elements of the row.

    That keeps `sizePerThread` high enough for the 1-D tile to lower into
    vectorized (rather than scalar) loads, while staying inside the
    131072-register-per-block budget of the MTT S5000.
    """
    num_warps = 8 if tile_n <= 4096 else 16
    return num_warps, 2


def _use_inner_fused_path(N):
    """Route medium-N inner rows through the single-pass fused kernel.

    For N <= 1024 the upstream inner kernel already satisfies
    ONE_TILE_PER_CTA and is single-pass, so it must be left alone.
    For N >= 16384 the row no longer fits in registers and the streaming
    large-N kernel is the better choice.
    """
    if _V11_DISABLED:
        return False
    if N <= _V11_MIN_N:
        return False
    if N > _V11_MAX_N:
        return False
    return True


@libentry()
@triton.jit
def softmax_backward_inner_large_n_kernel(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
):
    pid_m = ext.program_id(0)
    input_base = pid_m * N
    out_base = pid_m * N

    scale = tl.zeros([TILE_N], dtype=tl.float32)
    n_offsets = tl.arange(0, TILE_N)

    for _ in range(0, N, TILE_N):
        mask = n_offsets < N
        out_tile = tl.load(out_ptr + input_base + n_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        out_grad_tile = tl.load(
            out_grad_ptr + input_base + n_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        scale += out_tile * out_grad_tile
        n_offsets += TILE_N
    scale_sum = tl.sum(scale, axis=0)

    n_offsets = tl.arange(0, TILE_N)
    for _ in range(0, N, TILE_N):
        mask = n_offsets < N
        out_tile = tl.load(out_ptr + input_base + n_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        out_grad_tile = tl.load(
            out_grad_ptr + input_base + n_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        in_grad_tile = out_tile * (out_grad_tile - scale_sum)
        tl.store(
            in_grad_ptr + out_base + n_offsets,
            in_grad_tile.to(in_grad_ptr.dtype.element_ty),
            mask=mask,
        )
        n_offsets += TILE_N

def _non_inner_occ_configs():
    return [
        triton.Config(
            {"TILE_N": tile_n, "TILE_K": tile_k},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for tile_n, tile_k in (
            (16, 128),
            (16, 64),
            (16, 32),
            (16, 16),
            (32, 64),
            (32, 32),
            (32, 16),
            (64, 32),
            (64, 16),
        )
        for num_warps in (4,)
        for num_stages in (2, 3)
    ]

@libentry()
@triton.autotune(
    configs=_non_inner_occ_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def softmax_backward_non_inner_occ_kernel(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
):
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)

    out_ptr += pid_m.to(tl.int64) * N * K
    out_grad_ptr += pid_m.to(tl.int64) * N * K
    in_grad_ptr += pid_m.to(tl.int64) * N * K

    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)
    k_mask = k_offsets < K

    n_offsets = tl.arange(0, TILE_N)
    offsets = n_offsets[:, None] * K + k_offsets[None, :]
    acc = tl.zeros([TILE_N, TILE_K], dtype=tl.float32)

    for _ in range(0, N, TILE_N):
        mask = (n_offsets < N)[:, None] & k_mask[None, :]
        out_tile = tl.load(
            out_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_last"
        ).to(tl.float32)
        out_grad_tile = tl.load(
            out_grad_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_last"
        ).to(tl.float32)
        acc += out_tile * out_grad_tile
        offsets += TILE_N * K
        n_offsets += TILE_N

    scale_sum = tl.sum(acc, axis=0)

    n_offsets = tl.arange(0, TILE_N)
    offsets = n_offsets[:, None] * K + k_offsets[None, :]

    for _ in range(0, N, TILE_N):
        mask = (n_offsets < N)[:, None] & k_mask[None, :]
        out_tile = tl.load(
            out_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first"
        ).to(tl.float32)
        out_grad_tile = tl.load(
            out_grad_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first"
        ).to(tl.float32)
        in_grad_tile = out_tile * (out_grad_tile - scale_sum[None, :])
        tl.store(
            in_grad_ptr + offsets,
            in_grad_tile.to(in_grad_ptr.dtype.element_ty),
            mask=mask,
        )
        offsets += TILE_N * K
        n_offsets += TILE_N

def _use_non_inner_occ_path(M, N, K):
    if _V9_DISABLED:
        return False
    if K > 128:
        return False
    if N < 256:
        return False
    est_ctas = M * triton.cdiv(K, 64)
    return est_ctas < _NON_INNER_TARGET_CTAS


def _use_non_inner_decoupled_path(M, N, K):
    if _V10_DISABLED:
        return False
    if K < _DECOUPLED_MIN_K:
        return False
    if N < _DECOUPLED_MIN_N:
        return False
    _, upstream_tile_n = _upstream_backward_tile_n(M, N, K)
    return upstream_tile_n <= _DECOUPLED_PATHOLOGIC_TILE_N


def _decoupled_configs():
    tile_ks = (16, 32, 64, 128)
    tile_ns = (16, 32, 64, 128)
    configs = []
    for tile_k in tile_ks:
        for tile_n in tile_ns:
            elems = tile_n * tile_k
            if elems > _DECOUPLED_MAX_TILE_ELEMS:
                continue
            if elems < 2048:
                num_warps = 4
            elif elems < 4096:
                num_warps = 8
            else:
                num_warps = 16
            configs.append(
                triton.Config(
                    {"TILE_N": tile_n, "TILE_K": tile_k},
                    num_warps=num_warps,
                    num_stages=2,
                    num_ctas=1,
                )
            )
    return configs


@libentry()
@triton.autotune(
    configs=_decoupled_configs(),
    key=[
        "M",
        "N",
        "K",
    ],
)
@triton.jit
def softmax_backward_non_inner_decoupled_kernel(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
):
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)

    row_base = pid_m.to(tl.int64) * N * K
    out_ptr += row_base
    out_grad_ptr += row_base
    in_grad_ptr += row_base

    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)
    k_mask = k_offsets < K

    n_offsets = tl.arange(0, TILE_N)
    offsets = n_offsets[:, None] * K + k_offsets[None, :]
    acc = tl.zeros([TILE_N, TILE_K], dtype=tl.float32)

    for _ in range(0, N, TILE_N):
        mask = (n_offsets < N)[:, None] & k_mask[None, :]
        out_tile = tl.load(
            out_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_last"
        ).to(tl.float32)
        out_grad_tile = tl.load(
            out_grad_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_last"
        ).to(tl.float32)
        acc += out_tile * out_grad_tile
        offsets += TILE_N * K
        n_offsets += TILE_N

    scale_sum = tl.sum(acc, axis=0)

    n_offsets = tl.arange(0, TILE_N)
    offsets = n_offsets[:, None] * K + k_offsets[None, :]

    for _ in range(0, N, TILE_N):
        mask = (n_offsets < N)[:, None] & k_mask[None, :]
        out_tile = tl.load(
            out_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first"
        ).to(tl.float32)
        out_grad_tile = tl.load(
            out_grad_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first"
        ).to(tl.float32)
        in_grad_tile = out_tile * (out_grad_tile - scale_sum[None, :])
        tl.store(
            in_grad_ptr + offsets,
            in_grad_tile.to(in_grad_ptr.dtype.element_ty),
            mask=mask,
        )
        offsets += TILE_N * K
        n_offsets += TILE_N


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("softmax_non_inner"),
    key=[
        "M",
        "N",
        "K",
    ],
)
@triton.heuristics(runtime.get_heuristic_config("softmax_backward_non_inner"))
@triton.jit
def softmax_backward_kernel_non_inner(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    offsets_k = pid_k * TILE_K + tl.arange(0, TILE_K)

    if ONE_TILE_PER_CTA:
        offsets_n = tl.arange(0, TILE_N)
        offsets = pid_m * N * K + offsets_n[:, None] * K + offsets_k
        mask = (offsets_n < N)[:, None] & (offsets_k < K)
        out_tile = tl.load(out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.sum(out_tile * out_grad_tile, axis=0)
        in_grad_tile = out_tile * (out_grad_tile - scale[None, :])
        tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
    else:
        offsets_n = tl.arange(0, TILE_N)
        offsets = pid_m * N * K + offsets_n[:, None] * K + offsets_k
        scale = tl.zeros([TILE_N, TILE_K], dtype=tl.float32)
        for _ in range(0, N, TILE_N):
            mask = (offsets_n < N)[:, None] & (offsets_k < K)
            out_tile = tl.load(out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            out_grad_tile = tl.load(
                out_grad_ptr + offsets, mask=mask, other=0.0
            ).to(tl.float32)
            scale += out_tile * out_grad_tile
            offsets_n += TILE_N
            offsets += TILE_N * K
        scale_sum = tl.sum(scale, axis=0)

        offsets_n = tl.arange(0, TILE_N)
        offsets = pid_m * N * K + offsets_n[:, None] * K + offsets_k
        for _ in range(0, N, TILE_N):
            mask = (offsets_n < N)[:, None] & (offsets_k < K)
            out_tile = tl.load(out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            out_grad_tile = tl.load(
                out_grad_ptr + offsets, mask=mask, other=0.0
            ).to(tl.float32)
            in_grad_tile = out_tile * (out_grad_tile - scale_sum[None, :])
            tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
            offsets_n += TILE_N
            offsets += TILE_N * K

@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("softmax_inner"),
    key=["M", "N"],
)
@triton.heuristics(
    values=runtime.get_heuristic_config("softmax_backward_inner"),
)
@triton.jit
def softmax_backward_kernel_inner(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = ext.program_id(0)
    m_offsets = pid_m * TILE_M + tl.arange(0, TILE_M)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * N + n_offsets
        mask = (m_offsets[:, None] < M) & (n_offsets < N)
        out_tile = tl.load(out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.sum(out_tile * out_grad_tile, 1)
        in_grad_tile = out_tile * (out_grad_tile - scale[:, None])
        tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
    else:
        scale = tl.zeros([TILE_M, TILE_N], dtype=tl.float32)

        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * N + n_offsets
        for _ in range(0, N, TILE_N):
            mask = (m_offsets[:, None] < M) & (n_offsets < N)
            out_tile = tl.load(
                out_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_last"
            ).to(tl.float32)
            out_grad_tile = tl.load(
                out_grad_ptr + offsets, mask=mask, other=0.0
            ).to(tl.float32)
            scale += out_tile * out_grad_tile
            n_offsets += TILE_N
            offsets += TILE_N
        scale_sum = tl.sum(scale, 1)

        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * N + n_offsets
        for _ in range(0, N, TILE_N):
            mask = (m_offsets[:, None] < M) & (n_offsets < N)
            out_tile = tl.load(
                out_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first"
            ).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            in_grad_tile = out_tile * (out_grad_tile - scale_sum[:, None])
            tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
            n_offsets += TILE_N
            offsets += TILE_N

def softmax_out(self, dim, half_to_float=False, *, out):
    logger.debug("GEMS SOFTMAX_OUT")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"

    if self.numel() == 0:
        if tuple(out.shape) != tuple(self.shape):
            out.resize_(self.shape)
        zero_(out)
        return out

    dim = dim % self.ndim
    M = 1
    N = self.shape[dim]
    for i in range(dim):
        M *= self.shape[i]
    self = self.contiguous()
    dtype = torch.float32 if half_to_float else self.dtype
    if tuple(out.shape) != tuple(self.shape):
        out.resize_(self.shape)
    if out.dtype != dtype:
        raise RuntimeError(
            f"_softmax.out: expected out dtype {dtype}, got {out.dtype}"
        )
    K = self.numel() // M // N

    with torch_device_fn.device(self.device):
        if K > 1:
            grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
            softmax_kernel_non_inner[grid](
                out,
                self,
                M,
                N,
                K,
            )
        else:
            grid = (M, 1, 1)
            softmax_kernel_inner[grid](
                out,
                self,
                M,
                N,
            )
    return out

def softmax(self, dim, half_to_float=False):
    logger.debug("GEMS SOFTMAX")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"

    if self.numel() == 0:
        out_shape = list(self.shape)
        out = torch.empty(out_shape, dtype=self.dtype, device=self.device)
        zero_(out)
        return out

    dtype = torch.float32 if half_to_float else self.dtype
    out = torch.empty_like(self, dtype=dtype)
    return softmax_out(self, dim, half_to_float, out=out)

def softmax_backward_out(grad_output, output, dim, input_dtype, *, grad_input):
    logger.debug("GEMS SOFTMAX_BACKWARD_OUT")

    assert dim >= -output.ndim and dim < output.ndim, "Invalid dim"

    if output.numel() == 0:
        if tuple(grad_input.shape) != tuple(output.shape):
            grad_input.resize_(output.shape)
        zero_(grad_input)
        return grad_input

    dim = dim % output.ndim
    M = 1
    N = output.shape[dim]
    for i in range(dim):
        M *= output.shape[i]

    grad_output = grad_output.contiguous()
    if tuple(grad_input.shape) != tuple(output.shape):
        grad_input.resize_(output.shape)
    if grad_input.dtype != input_dtype:
        raise RuntimeError(
            f"_softmax_backward_data.out: expected grad_input dtype {input_dtype}, got {grad_input.dtype}"
        )
    K = output.numel() // M // N

    with torch_device_fn.device(grad_input.device):
        if K > 1:
            if _use_non_inner_occ_path(M, N, K):
                grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
                softmax_backward_non_inner_occ_kernel[grid](
                    output,
                    grad_output,
                    grad_input,
                    M,
                    N,
                    K,
                )
                return grad_input

            if _use_non_inner_decoupled_path(M, N, K):
                grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
                softmax_backward_non_inner_decoupled_kernel[grid](
                    output,
                    grad_output,
                    grad_input,
                    M,
                    N,
                    K,
                )
                return grad_input

            grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
            softmax_backward_kernel_non_inner[grid](
                output,
                grad_output,
                grad_input,
                M,
                N,
                K,
            )
        else:
            if M == 1 and N >= 65536:
                return softmax_backward_1d_fast(grad_output, output, grad_input)

            if _use_inner_fused_path(N):
                tile_n = _next_pow2(N)
                num_warps, num_stages = _inner_fused_launch(tile_n)
                grid = (M, 1, 1)
                softmax_backward_inner_fused_kernel[grid](
                    output,
                    grad_output,
                    grad_input,
                    M,
                    N,
                    TILE_N=tile_n,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
                return grad_input

            if N >= 16384:
                grid = (M, 1, 1)
                softmax_backward_inner_large_n_kernel[grid](
                    output,
                    grad_output,
                    grad_input,
                    M,
                    N,
                    TILE_N=4096,
                    num_warps=8,
                    num_stages=3,
                )
                return grad_input

            if N <= 32:
                tile_m, tile_n = _packed_small_n_tiles(N)
                grid = (triton.cdiv(M, tile_m), 1, 1)
                softmax_backward_inner_packed_small_n_kernel[grid](
                    output,
                    grad_output,
                    grad_input,
                    M,
                    N,
                    TILE_M=tile_m,
                    TILE_N=tile_n,
                    num_warps=4,
                )
                return grad_input

            grid = lambda meta: (triton.cdiv(M, meta["TILE_M"]), 1, 1)
            softmax_backward_kernel_inner[grid](
                output,
                grad_output,
                grad_input,
                M,
                N,
            )
    return grad_input

def softmax_backward(grad_output, output, dim, input_dtype):
    logger.debug("GEMS SOFTMAX_BACKWARD")
    in_grad = torch.empty_like(output, dtype=input_dtype)
    return softmax_backward_out(
        grad_output, output, dim, input_dtype, grad_input=in_grad
    )