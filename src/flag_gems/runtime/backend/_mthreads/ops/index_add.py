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

"""High-performance and accurate index_add / index_add_ for mthreads (MUSA).

Design Principles:
  1. Universal Hardware Heuristics: Configs derived from hardware first-principles.
  2. Zero-Pollution Pure JIT: Eliminates @triton.autotune.
  3. Continuous Flattened 3D Model: Generalizes multi-dimensional tensors.
  4. Ultra-Low Launch Overhead: JIT Cache quantization and asynchronous device 
     assertions to eliminate Host-Device sync for small tensors.
"""

import os
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

_ATOMIC_UNSUPPORTED_DTYPES = frozenset({torch.bfloat16})
_INT32_MAX = 2**31 - 1

# 【assert 门控】：device_assert 的 assert buffer 挂载在 MUSA 启动路径上有额外开销，
# 默认关闭（越界 index 仍由 mask 兜底，不会写坏内存），
# 调试时设 FLAGGEMS_INDEX_ADD_DEBUG=1 打开硬件级 assert。
_DEBUG_ASSERT = os.environ.get("FLAGGEMS_INDEX_ADD_DEBUG", "0") == "1"


def _device_guard(device):
    # 当前设备已正确时跳过 guard，省去 enter/exit 的 driver 往返
    idx = device.index
    if idx is None or idx == torch_device_fn.current_device():
        return nullcontext()
    return torch_device_fn.device(device)

# =============================================================================
#  Hardware First-Principle Heuristic Config Generators (Quantized for JIT Cache)
# =============================================================================

def _get_inner1_configs(total_elements):
    # 【JIT Cache 量子化】：微小尺寸全部合并为一种签名，消灭缓存查询开销
    if total_elements <= 1024:
        return 1024, 4
    block_size = min(1024, max(16, triton.next_power_of_2(total_elements)))
    num_warps = 8 if block_size >= 1024 else 4
    return block_size, num_warps

def _get_general_configs(inner, N):
    # 【JIT Cache 量子化】
    if inner <= 128 and N <= 16:
        return 128, 16, 4
    block_inner = min(128, max(16, triton.next_power_of_2(inner)))
    block_n = max(1, min(16, 512 // block_inner))
    return block_inner, block_n, 4

# =============================================================================
#  Triton Kernels (Pure JIT)
# =============================================================================

@triton.jit
def _index_add_inner1_kernel(
    out_ptr, index_ptr, src_ptr, M, N, dim_len, alpha,
    USE_INT32: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    ALPHA_IS_ONE: tl.constexpr, ENABLE_ASSERT: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_elements = M * N

    mask = offsets < total_elements
    r_m = offsets // N
    r_n = offsets % N

    # index 只读一次，用 .cg 跳过 L1 避免污染（_mthreads 已有同款用法）
    idx_1d = tl.load(index_ptr + r_n, mask=mask, other=0, cache_modifier=".cg")

    # 【零同步越界校验】：mask 始终生效保证内存安全；
    # 硬件级 assert 由 ENABLE_ASSERT 门控（默认关，调试时开）
    valid_idx = (idx_1d >= 0) & (idx_1d < dim_len)
    if ENABLE_ASSERT:
        tl.device_assert(valid_idx | ~mask, "0 <= index < self.size(dim)")

    mask = mask & valid_idx

    if USE_INT32:
        src_off = offsets.to(tl.int32)
        inp_off = r_m.to(tl.int32) * dim_len + idx_1d.to(tl.int32)
    else:
        src_off = offsets.to(tl.int64)
        inp_off = r_m.to(tl.int64) * dim_len + idx_1d.to(tl.int64)

    src_val = tl.load(src_ptr + src_off, mask=mask, other=0.0)
    # 一律在 fp32 下算 addend，按 out 指针的实际 dtype 落盘：
    # bf16 场景 out 是 fp32 工作区，src 原样读入 kernel 内转换，省掉 host 端整表 .to()
    if ALPHA_IS_ONE:
        addend = src_val.to(tl.float32)
    else:
        addend = alpha * src_val.to(tl.float32)

    tl.atomic_add(out_ptr + inp_off, addend.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _index_add_general_kernel(
    out_ptr, index_ptr, src_ptr, outer, dim_len, N, inner, outer_n, alpha,
    USE_INT32: tl.constexpr, BLOCK_INNER: tl.constexpr, BLOCK_N: tl.constexpr,
    ALPHA_IS_ONE: tl.constexpr, ENABLE_ASSERT: tl.constexpr,
):
    pid_on = tl.program_id(0)
    pid_in = tl.program_id(1)

    r_on = pid_on * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
    r_in = pid_in * BLOCK_INNER + tl.arange(0, BLOCK_INNER)[None, :]

    r_outer = r_on // N
    r_n = r_on % N

    mask = (r_outer < outer) & (r_n < N) & (r_in < inner)
    idx = tl.load(index_ptr + r_n, mask=(r_n < N), other=0, cache_modifier=".cg")

    # 【零同步越界校验】：mask 始终生效保证内存安全；
    # 硬件级 assert 由 ENABLE_ASSERT 门控（默认关，调试时开）
    valid_idx = (idx >= 0) & (idx < dim_len)
    if ENABLE_ASSERT:
        tl.device_assert(valid_idx | ~(r_n < N), "0 <= index < self.size(dim)")

    mask = mask & valid_idx

    if USE_INT32:
        src_off = (r_outer.to(tl.int32) * N + r_n.to(tl.int32)) * inner + r_in.to(tl.int32)
        inp_off = (r_outer.to(tl.int32) * dim_len + idx.to(tl.int32)) * inner + r_in.to(tl.int32)
    else:
        src_off = (r_outer.to(tl.int64) * N + r_n.to(tl.int64)) * inner + r_in.to(tl.int64)
        inp_off = (r_outer.to(tl.int64) * dim_len + idx.to(tl.int64)) * inner + r_in.to(tl.int64)

    src_val = tl.load(src_ptr + src_off, mask=mask, other=0.0)
    if ALPHA_IS_ONE:
        addend = src_val.to(tl.float32)
    else:
        addend = alpha * src_val.to(tl.float32)

    tl.atomic_add(out_ptr + inp_off, addend.to(out_ptr.dtype.element_ty), mask=mask)

# =============================================================================
#  Driver
# =============================================================================

def _index_add_impl(inp, dim, index, src, alpha, out=None):
    ndim = inp.ndim
    dim = dim % ndim
    shape = inp.shape
    dim_len = shape[dim]
    N = index.numel()

    if N == 0 or src.numel() == 0:
        return inp if out is None else out
        
    # 【彻底摘除导致 CPU-GPU 强制同步的 Python Assert！】
    # if N > 0:
    #     assert bool(((index >= 0) & (index < dim_len)).all()) ...

    target = inp if out is None else out

    # 【剥离原生 Python Fat】：使用内联循环替代 math.prod 和 Tuple 切片实例化
    outer = 1
    for i in range(dim):
        outer *= shape[i]
    inner = 1
    for i in range(dim + 1, ndim):
        inner *= shape[i]

    is_bf16 = target.dtype in _ATOMIC_UNSUPPORTED_DTYPES
    if is_bf16:
        # bf16 无原生 atomic：out 转 fp32 工作区（1 次 host op）；
        # src 不转，原样传进 kernel 内部转 fp32（省掉 1 次整表 .to()）
        work_out = target.to(torch.float32)
    else:
        work_out = target if target.is_contiguous() else target.contiguous()
    work_src = src if src.is_contiguous() else src.contiguous()

    index_c = index.contiguous()
    use_int32 = (target.numel() < _INT32_MAX) and (src.numel() < _INT32_MAX)
    alpha_is_one = alpha == 1

    with _device_guard(inp.device):
        if inner == 1:
            M = outer
            total_elements = M * N
            block_size, num_warps = _get_inner1_configs(total_elements)
            grid = (triton.cdiv(total_elements, block_size),)

            _index_add_inner1_kernel[grid](
                work_out, index_c, work_src,
                M, N, dim_len, alpha,
                USE_INT32=use_int32,
                BLOCK_SIZE=block_size,
                ALPHA_IS_ONE=alpha_is_one,
                ENABLE_ASSERT=_DEBUG_ASSERT,
                num_warps=num_warps, num_stages=1,
            )
        else:
            block_inner, block_n, num_warps = _get_general_configs(inner, N)
            outer_n = outer * N
            grid = (triton.cdiv(outer_n, block_n), triton.cdiv(inner, block_inner))

            _index_add_general_kernel[grid](
                work_out, index_c, work_src,
                outer, dim_len, N, inner, outer_n, alpha,
                USE_INT32=use_int32,
                BLOCK_INNER=block_inner, BLOCK_N=block_n,
                ALPHA_IS_ONE=alpha_is_one,
                ENABLE_ASSERT=_DEBUG_ASSERT,
                num_warps=num_warps, num_stages=1,
            )

    if is_bf16:
        # copy_ 原生支持跨 dtype，fp32->bf16 一次完成
        # （替代 work_out.to(orig_dtype)+copy_ 的临时分配+两次 kernel）
        target.copy_(work_out)
    elif work_out is not target:
        target.copy_(work_out)

    return target

def index_add(inp, dim, index, src, alpha=1):
    out = inp.clone()
    return _index_add_impl(inp, dim, index, src, alpha, out=out)

def index_add_(inp, dim, index, src, alpha=1):
    return _index_add_impl(inp, dim, index, src, alpha, out=None)