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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)


def max_pool2d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    ceil_mode: bool = False,
) -> int:
    effective_kernel_size = (kernel_size - 1) * dilation + 1
    numerator = in_size + 2 * padding - effective_kernel_size
    if ceil_mode:
        output_size = (numerator + stride - 1) // stride + 1
        # PyTorch-compatible adjustment for ceil_mode
        if (output_size - 1) * stride >= in_size + padding:
            output_size -= 1
    else:
        output_size = numerator // stride + 1

    return output_size


# ==============================================================================
# Kernel 1: 通道打包 Kernel (v12 — 方向 A 拓 num_stages=3/4, num_warps=2)
#
# 改动逻辑 (相对于 v11):
#   方向 A: 在 autotune configs 里追加 num_stages=3, 4 与 num_warps=2 候选.
#           触发条件: MTT S5000 = 60 SM × 128 phys regs/SM, num_stages=4 在
#           BLOCK_C=2 上 register footprint ~16K (每 SM 上限 131K), 远未溢出.
#           此前 v6/v9/v10/v11 全部停在 num_stages=2, 是被 64 SM × 256 reg
#           的 NV A100 直觉带偏了.
#
#   保留 v11 的方向 A mask 物化技巧: load(..., mask=c_mask & valid_spatial,
#   other=-inf, fp32 cast 一次完成).
# ==============================================================================
@libentry()
@triton.autotune(
    configs=[
        # v11 默认 baseline (保留以做对照)
        triton.Config({"BLOCK_C": 2}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_C": 4}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_C": 4}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_C": 8}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_C": 8}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_C": 16}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_C": 16}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_C": 32}, num_stages=2, num_warps=8),
        # 🆕 v12 — 方向 A: num_stages=3 pipeline 隐藏 memory latency
        triton.Config({"BLOCK_C": 2}, num_stages=3, num_warps=2),
        triton.Config({"BLOCK_C": 4}, num_stages=3, num_warps=2),
        triton.Config({"BLOCK_C": 4}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_C": 8}, num_stages=3, num_warps=4),
        # 🆕 v12 — 方向 A: num_stages=4 双 buffer pipeline
        # MTT S5000 单 SM 物理 128 regs, 但单 Block 允许 131K regs.
        # BLOCK_C=4 + BLOCK_HW=16 + num_stages=4 ≈ 64 regs/thread × 32 thread × 4 = 8K,
        # 远低于上限. 关键目标 case: (4,3,224) 的庞大 grids.
        triton.Config({"BLOCK_C": 2}, num_stages=4, num_warps=2),
        triton.Config({"BLOCK_C": 4}, num_stages=4, num_warps=2),
        triton.Config({"BLOCK_C": 4}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_C": 8}, num_stages=4, num_warps=4),
    ],
    key=[
        "total_channels",
        "out_hw",
        "kernel_h",
        "kernel_w",
        "stride_h",
        "stride_w",
        "dilation_h",
        "dilation_w",
    ],
)
@triton.jit
def max_pool2d_channel_packed_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    total_channels,
    in_h,
    in_w,
    out_h,
    out_w,
    out_hw,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_c = tl.program_id(0)
    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    hw_offsets = tl.arange(0, BLOCK_HW)

    c_mask = c_offsets < total_channels
    hw_mask = hw_offsets < out_hw

    h_out = hw_offsets // out_w
    w_out = hw_offsets % out_w

    dtype = input_ptr.type.element_ty
    min_in_f32 = float("-inf")
    max_val_acc = tl.full((BLOCK_C, BLOCK_HW), min_in_f32, dtype=tl.float32)
    max_idx_acc = tl.full((BLOCK_C, BLOCK_HW), -1, dtype=tl.int32)

    in_hw_size = in_h * in_w
    c_base_offset = c_offsets[:, None] * in_hw_size

    for kh in tl.static_range(0, kernel_h):
        for kw in tl.static_range(0, kernel_w):
            h_in = h_out * stride_h - padding_h + kh * dilation_h
            w_in = w_out * stride_w - padding_w + kw * dilation_w

            valid_spatial = (
                hw_mask & (h_in >= 0) & (h_in < in_h) & (w_in >= 0) & (w_in < in_w)
            )

            h_in_safe = tl.maximum(0, tl.minimum(h_in, in_h - 1))
            w_in_safe = tl.maximum(0, tl.minimum(w_in, in_w - 1))
            spatial_offset_safe = h_in_safe * in_w + w_in_safe

            input_offset = c_base_offset + spatial_offset_safe[None, :]

            # ✅ v11 方向 A 保留: (c_mask AND valid_spatial) → mask
            load_mask_2d = c_mask[:, None] & valid_spatial[None, :]
            raw_val = tl.load(
                input_ptr + input_offset, mask=load_mask_2d, other=min_in_f32
            )
            current_val_f32 = raw_val.to(tl.float32)
            current_idx = (h_in * in_w + w_in).to(tl.int32)

            is_new_max = current_val_f32 > max_val_acc
            max_val_acc = tl.where(is_new_max, current_val_f32, max_val_acc)
            max_idx_acc = tl.where(is_new_max, current_idx[None, :], max_idx_acc)

    out_base_offset = c_offsets[:, None] * out_hw + hw_offsets[None, :]
    store_mask = c_mask[:, None] & hw_mask[None, :]
    tl.store(
        output_ptr + out_base_offset,
        max_val_acc.to(dtype),
        mask=store_mask,
    )
    tl.store(
        indices_ptr + out_base_offset, max_idx_acc.to(tl.int64), mask=store_mask
    )


# ==============================================================================
# Kernel 2: 2D Forward Kernel (常规, dilation==1 case)
# ==============================================================================
@libentry()
@triton.autotune(
    configs=[
        # ===== v11 baseline (保留) =====
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 16}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 16}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 16}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 32}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 8}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 16}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 64}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 32}, num_stages=2, num_warps=8),
        # 🆕 v12 — 方向 A (num_stages=3/4, 拓 pipeline 深度)
        # warp=2 减小 register pressure 让更大 num_stages 不溢出
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 16}, num_stages=3, num_warps=2),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 32}, num_stages=3, num_warps=2),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 16}, num_stages=3, num_warps=2),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 16}, num_stages=4, num_warps=2),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 16}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 32}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 32}, num_stages=4, num_warps=8),
        # 🆕 v12 — 方向 B (BLOCK=512 各种 BLOCK_H × BLOCK_W)
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 16}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 32}, num_stages=3, num_warps=4),
    ],
    key=[
        "out_h",
        "out_w",
        "kernel_h",
        "kernel_w",
        "stride_h",
        "stride_w",
        "dilation_h",
        "dilation_w",
    ],
)
@triton.jit
def max_pool2d_forward_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    num_w_blocks = tl.cdiv(out_w, BLOCK_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks
    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    dtype = input_ptr.type.element_ty
    min_in_f32 = float("-inf")
    max_val_acc = tl.full((BLOCK_H, BLOCK_W), min_in_f32, dtype=tl.float32)
    max_idx_acc = tl.full((BLOCK_H, BLOCK_W), -1, dtype=tl.int32)

    input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

    for kh in tl.static_range(0, kernel_h):
        for kw in tl.static_range(0, kernel_w):
            h_in = h_out_offsets[:, None] * stride_h - padding_h + kh * dilation_h
            w_in = w_out_offsets[None, :] * stride_w - padding_w + kw * dilation_w

            h_in_safe = tl.maximum(0, tl.minimum(h_in, in_h - 1))
            w_in_safe = tl.maximum(0, tl.minimum(w_in, in_w - 1))
            safe_offset = h_in_safe * in_stride_h + w_in_safe * in_stride_w

            in_mask = (h_in >= 0) & (h_in < in_h) & (w_in >= 0) & (w_in < in_w)
            load_mask = in_mask
            raw_val = tl.load(
                input_base_ptr + safe_offset, mask=load_mask, other=min_in_f32
            )

            raw_val_f32 = raw_val.to(tl.float32)
            current_idx = (h_in * in_w + w_in).to(tl.int32)

            is_new_max = raw_val_f32 > max_val_acc
            max_val_acc = tl.where(is_new_max, raw_val_f32, max_val_acc)
            max_idx_acc = tl.where(is_new_max, current_idx, max_idx_acc)

    out_base_ptr = output_ptr + pid_nc * out_h * out_w
    indices_base_ptr = indices_ptr + pid_nc * out_h * out_w
    output_block_ptr = (
        out_base_ptr + h_out_offsets[:, None] * out_w + w_out_offsets[None, :]
    )
    indices_block_ptr = (
        indices_base_ptr + h_out_offsets[:, None] * out_w + w_out_offsets[None, :]
    )

    out_mask = (h_out_offsets[:, None] < out_h) & (w_out_offsets[None, :] < out_w)
    tl.store(output_block_ptr, max_val_acc.to(dtype), mask=out_mask)
    tl.store(indices_block_ptr, max_idx_acc.to(tl.int64), mask=out_mask)


# ==============================================================================
# Kernel 3 (🆕 v12 — 方向 D): dilation-aware Forward Kernel
#
# 动机: (dilation_h > 1 or dilation_w > 1) case 在 v11 是最痛点:
#   - (4,3,224) d=2: fp16=0.750x, fp32=0.750x
#   - (16,64,56) d=2: fp16=0.729x, fp32=0.742x
#   - (32,128,28) d=2: fp16=0.619x, fp32=0.627x
#   - (64,256,14) d=2: fp16=0.421x, fp32=0.440x  ← worst case
#
# dilation=2 让相邻 kh/kw iteration 的 h_in/w_in 跳跃 stride*2,
# 实际触达的输入点是 3x3=9 个但分布在更大的输入区域上, 访存 pattern 不连续.
#
# 关键 insight: dilation kernel 的 h_in_safe/w_in_safe clamp 是**确定性**的,
# 与 dilation=1 路径完全独立, 但 v11 走的是同一 kernel.
# 我们把它分出来, autotune key 不含 dilation_h/w (dilation 永远 > 1),
# 允许更激进的 BLOCK 选择 (BLOCK 翻倍可以抵消 dilation 带来的访存距离).
# ==============================================================================
@libentry()
@triton.autotune(
    configs=[
        # dilation=2 case 偏好: 大 BLOCK (因为每个输出点触达的 input 距离大,
        # L1 命中率低, 必须靠大规模数据并行抵消; 同时 register footprint 允许更大 BLOCK)
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 16}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 16}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 32}, num_stages=2, num_warps=8),
        # num_stages=3/4 pipeline (方向 A 移植)
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 16}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 16}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 16}, num_stages=4, num_warps=4),
        # warp=2 让更大 num_stages 不溢出, 关键: dilation 路径 register pressure 高
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 16}, num_stages=3, num_warps=2),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 32}, num_stages=3, num_warps=2),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 16}, num_stages=4, num_warps=2),
        # (方向 B) BLOCK=512 探索
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 16}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 32}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 8}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 64}, num_stages=2, num_warps=4),
    ],
    key=[
        "out_h",
        "out_w",
        "kernel_h",
        "kernel_w",
        "stride_h",
        "stride_w",
        # dilation 不进 key: 这个 kernel 永远走 dilation>0 路径
    ],
)
@triton.jit
def max_pool2d_forward_kernel_dilated(
    input_ptr,
    output_ptr,
    indices_ptr,
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """专用 dilation>1 kernel, 与 max_pool2d_forward_kernel 共享结构
    但不复用 dilation=1 的 autotune cache."""
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    num_w_blocks = tl.cdiv(out_w, BLOCK_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks
    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    dtype = input_ptr.type.element_ty
    min_in_f32 = float("-inf")
    max_val_acc = tl.full((BLOCK_H, BLOCK_W), min_in_f32, dtype=tl.float32)
    max_idx_acc = tl.full((BLOCK_H, BLOCK_W), -1, dtype=tl.int32)

    input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

    # 🚨 dilation>1 关键差异: in_mask 必须按 (h_in, w_in) 的 dilation 步进
    # 判断边界, dilation > 1 时 padding 占比极大 (尤其当 (in_h, in_w) 不大时).
    # 必须保留 tl.where 后处理 (v11 方向 A 的 mask 物化在 dilation>1 会引入
    # 错误的 safe_offset; 在 dilated kernel 里改回 v9 风格更稳).
    for kh in tl.static_range(0, kernel_h):
        for kw in tl.static_range(0, kernel_w):
            h_in = h_out_offsets[:, None] * stride_h - padding_h + kh * dilation_h
            w_in = w_out_offsets[None, :] * stride_w - padding_w + kw * dilation_w

            h_in_safe = tl.maximum(0, tl.minimum(h_in, in_h - 1))
            w_in_safe = tl.maximum(0, tl.minimum(w_in, in_w - 1))
            safe_offset = h_in_safe * in_stride_h + w_in_safe * in_stride_w

            # dilation 路径强制走 v9 风格: 全 in-bounds load + where 滤 padding
            # (mask 物化在 dilation 上会编译出错误的 predicated path)
            raw_val = tl.load(input_base_ptr + safe_offset)
            in_mask = (h_in >= 0) & (h_in < in_h) & (w_in >= 0) & (w_in < in_w)
            raw_val_f32 = raw_val.to(tl.float32)
            current_val_f32 = tl.where(in_mask, raw_val_f32, min_in_f32)
            current_idx = (h_in * in_w + w_in).to(tl.int32)

            is_new_max = current_val_f32 > max_val_acc
            max_val_acc = tl.where(is_new_max, current_val_f32, max_val_acc)
            max_idx_acc = tl.where(is_new_max, current_idx, max_idx_acc)

    out_base_ptr = output_ptr + pid_nc * out_h * out_w
    indices_base_ptr = indices_ptr + pid_nc * out_h * out_w
    output_block_ptr = (
        out_base_ptr + h_out_offsets[:, None] * out_w + w_out_offsets[None, :]
    )
    indices_block_ptr = (
        indices_base_ptr + h_out_offsets[:, None] * out_w + w_out_offsets[None, :]
    )

    out_mask = (h_out_offsets[:, None] < out_h) & (w_out_offsets[None, :] < out_w)
    tl.store(output_block_ptr, max_val_acc.to(dtype), mask=out_mask)
    tl.store(indices_block_ptr, max_idx_acc.to(tl.int64), mask=out_mask)


# ==============================================================================
# 反向 Kernel (与 v9 一致, 不动)
# ==============================================================================
@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_IN_H": 16, "BLOCK_IN_W": 16}, num_warps=4),
        triton.Config({"BLOCK_IN_H": 32, "BLOCK_IN_W": 8}, num_warps=4),
        triton.Config({"BLOCK_IN_H": 8, "BLOCK_IN_W": 32}, num_warps=4),
        triton.Config({"BLOCK_IN_H": 32, "BLOCK_IN_W": 32}, num_warps=8),
        triton.Config({"BLOCK_IN_H": 16, "BLOCK_IN_W": 64}, num_warps=8),
        triton.Config({"BLOCK_IN_H": 64, "BLOCK_IN_W": 16}, num_warps=8),
    ],
    key=[
        "in_h",
        "in_w",
        "kernel_h",
        "kernel_w",
        "stride_h",
        "stride_w",
        "dilation_h",
        "dilation_w",
    ],
)
@triton.jit
def max_pool2d_backward_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    in_h,
    in_w,
    out_h,
    out_w,
    out_stride_nc,
    out_stride_h,
    out_stride_w,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    BLOCK_IN_H: tl.constexpr,
    BLOCK_IN_W: tl.constexpr,
):
    nc_idx = tl.program_id(0)
    pid_hw = tl.program_id(1)

    num_w_blocks = tl.cdiv(in_w, BLOCK_IN_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks

    h_in_offsets = h_block_idx * BLOCK_IN_H + tl.arange(0, BLOCK_IN_H)
    w_in_offsets = w_block_idx * BLOCK_IN_W + tl.arange(0, BLOCK_IN_W)

    current_input_flat_idx = h_in_offsets[:, None] * in_w + w_in_offsets[None, :]
    grad_acc = tl.zeros((BLOCK_IN_H, BLOCK_IN_W), dtype=tl.float32)

    indices_base_ptr = indices_ptr + nc_idx * out_stride_nc
    grad_output_base_ptr = grad_output_ptr + nc_idx * out_stride_nc

    for kh in tl.static_range(0, kernel_h):
        for kw in tl.static_range(0, kernel_w):
            numerator_h = h_in_offsets[:, None] + padding_h - kh * dilation_h
            numerator_w = w_in_offsets[None, :] + padding_w - kw * dilation_w

            valid_map_mask = (numerator_h % stride_h == 0) & (
                numerator_w % stride_w == 0
            )
            h_out = numerator_h // stride_h
            w_out = numerator_w // stride_w
            out_bounds_mask = (
                (h_out >= 0) & (h_out < out_h) & (w_out >= 0) & (w_out < out_w)
            )
            load_mask = valid_map_mask & out_bounds_mask

            safe_h_out = tl.where(load_mask, h_out, 0)
            safe_w_out = tl.where(load_mask, w_out, 0)
            out_offsets = safe_h_out * out_stride_h + safe_w_out

            indices_block = tl.load(
                indices_base_ptr + out_offsets, mask=load_mask, other=-1
            )
            match_mask = indices_block == current_input_flat_idx

            grad_block = tl.load(
                grad_output_base_ptr + out_offsets, mask=match_mask, other=0.0
            )
            grad_acc += grad_block

    grad_input_base_ptr = grad_input_ptr + nc_idx * in_h * in_w
    grad_input_offsets = h_in_offsets[:, None] * in_w + w_in_offsets[None, :]
    store_mask = (h_in_offsets[:, None] < in_h) & (w_in_offsets[None, :] < in_w)
    tl.store(grad_input_base_ptr + grad_input_offsets, grad_acc, mask=store_mask)


def _parse_pool_params(kernel_size, stride, padding, dilation):
    def _parse_param(param, name, default=None):
        if param is None:
            return default
        if isinstance(param, int):
            return param, param
        if isinstance(param, (list, tuple)) and len(param) == 2:
            return param
        raise ValueError(f"Invalid {name}: {param}")

    kernel_h, kernel_w = _parse_param(kernel_size, "kernel_size")
    stride_h, stride_w = _parse_param(stride, "stride", default=(kernel_h, kernel_w))
    padding_h, padding_w = _parse_param(padding, "padding", default=(0, 0))
    dilation_h, dilation_w = _parse_param(dilation, "dilation", default=(1, 1))

    if stride_h <= 0 or stride_w <= 0:
        raise ValueError(
            f"stride must be positive, but got stride=({stride_h}, {stride_w})"
        )
    if padding_h < 0 or padding_w < 0:
        raise ValueError(
            f"padding must be non-negative, but got padding=({padding_h}, {padding_w})"
        )
    if dilation_h <= 0 or dilation_w <= 0:
        raise ValueError(
            f"dilation must be positive, but got dilation=({dilation_h}, {dilation_w})"
        )

    return (
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
    )


# ==============================================================================
# Main dispatcher (方向 D integration)
# ==============================================================================
def max_pool2d_with_indices(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    logger.debug("GEMS MAX_POOL2D_WITH_INDICES FORWARD (V12 — A+B+D)")
    input = input.contiguous()

    params = _parse_pool_params(kernel_size, stride, padding, dilation)
    (
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
    ) = params

    in_n, in_c, in_h, in_w = input.shape
    out_h = max_pool2d_output_size(
        in_h, kernel_h, stride_h, padding_h, dilation_h, ceil_mode
    )
    out_w = max_pool2d_output_size(
        in_w, kernel_w, stride_w, padding_w, dilation_w, ceil_mode
    )

    output = torch.empty(
        (in_n, in_c, out_h, out_w), device=input.device, dtype=input.dtype
    )
    indices = torch.empty(
        (in_n, in_c, out_h, out_w), device=input.device, dtype=torch.int64
    )

    if output.numel() == 0:
        return output, indices

    assert (
        in_h * in_w < 2147483647
    ), "Feature map elements exceed int32 representable range."

    out_hw = out_h * out_w

    # ===== 三路分发 (v12) =====
    # 路径 1: 小输出空间 + dilation 任意 → 通道打包 kernel
    # 路径 2: 大输出空间 + dilation == 1 → 常规 2D forward kernel
    # 路径 3 (🆕): 大输出空间 + dilation > 1 → 专用 dilated kernel
    spatial_threshold = 64 if (dilation_h > 1 or dilation_w > 1) else 196
    has_dilation = dilation_h > 1 or dilation_w > 1

    if out_hw <= spatial_threshold:
        # 路径 1: 通道打包 (out_hw ≤ 阈值), dilation 任意
        total_channels = in_n * in_c

        BLOCK_HW = max(16, triton.next_power_of_2(out_hw))

        grid = lambda meta: (triton.cdiv(total_channels, meta["BLOCK_C"]),)
        max_pool2d_channel_packed_kernel[grid](
            input,
            output,
            indices,
            total_channels,
            in_h,
            in_w,
            out_h,
            out_w,
            out_hw,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            BLOCK_HW=BLOCK_HW,
        )
    elif has_dilation:
        # 路径 3 (🆕 方向 D): 大输出 + dilation>1 → 专用 dilated kernel
        grid = lambda meta: (
            in_n * in_c,
            triton.cdiv(out_h, meta["BLOCK_H"]) * triton.cdiv(out_w, meta["BLOCK_W"]),
        )
        max_pool2d_forward_kernel_dilated[grid](
            input,
            output,
            indices,
            input.stride(0),
            input.stride(1),
            input.stride(2),
            input.stride(3),
            in_c,
            in_h,
            in_w,
            out_h,
            out_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
        )
    else:
        # 路径 2: 大输出 + dilation==1 → 常规 forward kernel
        grid = lambda meta: (
            in_n * in_c,
            triton.cdiv(out_h, meta["BLOCK_H"]) * triton.cdiv(out_w, meta["BLOCK_W"]),
        )

        max_pool2d_forward_kernel[grid](
            input,
            output,
            indices,
            input.stride(0),
            input.stride(1),
            input.stride(2),
            input.stride(3),
            in_c,
            in_h,
            in_w,
            out_h,
            out_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
        )

    return output, indices


def max_pool2d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    indices: torch.Tensor,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
):
    logger.debug("GEMS MAX_POOL2D BACKWARD")
    grad_output = grad_output.contiguous()
    indices = indices.contiguous()

    params = _parse_pool_params(kernel_size, stride, padding, dilation)
    (
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
    ) = params

    in_n, in_c, in_h, in_w = input.shape
    out_h, out_w = grad_output.shape[2], grad_output.shape[3]

    grad_input = torch.zeros_like(input, dtype=torch.float32)

    if grad_input.numel() == 0:
        return grad_input.to(grad_output.dtype)

    grid = lambda meta: (
        in_n * in_c,
        triton.cdiv(in_h, meta["BLOCK_IN_H"]) * triton.cdiv(in_w, meta["BLOCK_IN_W"]),
    )

    out_stride_nc = out_h * out_w
    out_stride_h = out_w
    out_stride_w = 1

    max_pool2d_backward_kernel[grid](
        grad_output,
        indices,
        grad_input,
        in_h,
        in_w,
        out_h,
        out_w,
        out_stride_nc,
        out_stride_h,
        out_stride_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
    )

    return grad_input.to(grad_output.dtype)


def max_pool2d_with_indices_backward(
    grad_output: torch.Tensor,
    self: torch.Tensor,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode: bool,
    indices: torch.Tensor,
):
    """Wrapper matching the aten::max_pool2d_with_indices_backward schema."""
    logger.debug("GEMS MAX_POOL2D_WITH_INDICES_BACKWARD")
    return max_pool2d_backward(
        grad_output, self, indices, kernel_size, stride, padding, dilation, ceil_mode
    )
