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
# See the License for the specific language without warranties or conditions
# of any kind, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

_REDUCE_MIN_M = 8192
_REDUCE_MIN_AVG_COUNT = 16
_ATOMIC_BLOCK_N = 128
_REDUCE_BLOCK_N = 128
_CSR_BLOCK = 256


@libentry()
@triton.jit
def embedding_kernel(
    out_ptr,
    in_ptr,
    weight_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    out_ptr += pid * N
    in_ptr += pid

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)

    row_idx = tl.load(in_ptr)
    weight_ptr += row_idx * N
    embedding_weight = tl.load(weight_ptr + cols, mask, other=0.0)
    tl.store(out_ptr + cols, embedding_weight, mask)


@libentry()
@triton.jit
def embedding_backward_bincount_kernel(
    indices_ptr,
    counts_ptr,
    M,
    BLOCK: tl.constexpr,
):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    idx = tl.load(indices_ptr + offs, mask=mask, other=0)
    tl.atomic_add(counts_ptr + idx, 1, mask=mask)


@libentry()
@triton.jit
def embedding_backward_exclusive_scan_kernel(counts_ptr, offsets_ptr, n):
    tl.store(offsets_ptr, 0)
    acc = 0
    for i in range(n):
        c = tl.load(counts_ptr + i)
        acc = acc + c
        tl.store(offsets_ptr + i + 1, acc)


@libentry()
@triton.jit
def embedding_backward_fill_rows_kernel(
    indices_ptr,
    sort_idx_ptr,
    cursor_ptr,
    M,
    BLOCK: tl.constexpr,
):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    idx = tl.load(indices_ptr + offs, mask=mask, other=0)
    slot = tl.atomic_add(cursor_ptr + idx, 1, mask=mask)
    tl.store(sort_idx_ptr + slot, offs.to(tl.int64), mask=mask)


def _build_csr(flat_indices, num_weights, device):
    M = flat_indices.numel()
    counts = torch.zeros(num_weights, device=device, dtype=torch.int32)
    offsets = torch.empty(num_weights + 1, device=device, dtype=torch.int32)
    sort_idx = torch.empty(M, device=device, dtype=torch.int64)
    grid = (triton.cdiv(M, _CSR_BLOCK),)

    with torch_device_fn.device(device):
        embedding_backward_bincount_kernel[grid](
            flat_indices, counts, M, _CSR_BLOCK
        )
        embedding_backward_exclusive_scan_kernel[(1,)](
            counts, offsets, num_weights
        )
        cursor = offsets[:-1].clone()
        embedding_backward_fill_rows_kernel[grid](
            flat_indices, sort_idx, cursor, M, _CSR_BLOCK
        )
    return sort_idx, offsets, counts


@libentry()
@triton.jit(do_not_specialize=["padding_idx"])
def embedding_backward_pipelined_reduce_kernel(
    out_ptr,
    grad_out,
    sort_idx,
    offsets,
    padding_idx,
    HAS_PADDING_IDX: tl.constexpr,
    SCALE_GRAD_BY_FREQ: tl.constexpr,
    N: tl.constexpr,
    N_VOCAB: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    N_COL_BLOCKS: tl.constexpr,
    N_M_BLOCKS: tl.constexpr,
):
    pid = ext.program_id(0)
    v = pid // (N_COL_BLOCKS * N_M_BLOCKS)
    rem = pid % (N_COL_BLOCKS * N_M_BLOCKS)
    col_block_id = rem // N_M_BLOCKS
    m_block_id = rem % N_M_BLOCKS

    if HAS_PADDING_IDX and v == padding_idx:
        return

    start = tl.load(offsets + v)
    end = tl.load(offsets + v + 1)
    count = end - start
    if count == 0:
        return

    seg_start = start + m_block_id * BLOCK_M
    if seg_start >= end:
        return

    col_offsets = col_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = col_offsets < N
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    for j in range(BLOCK_M):
        idx = seg_start + j
        valid = idx < end
        row_id = tl.load(sort_idx + idx, mask=valid, other=0)
        g = tl.load(
            grad_out + row_id * N + col_offsets,
            mask=col_mask & valid,
            other=0.0,
        )
        acc += g.to(tl.float32)

    if SCALE_GRAD_BY_FREQ:
        scale = tl.where(count > 1, 1.0 / count.to(tl.float32), 1.0)
        acc = acc * scale

    dst = (
        out_ptr
        + m_block_id * N_VOCAB * N
        + v * N
        + col_offsets
    )
    tl.store(dst, acc.to(out_ptr.dtype.element_ty), mask=col_mask)


@libentry()
@triton.jit
def embedding_backward_merge_partials_kernel(
    grad_in,
    partials,
    N: tl.constexpr,
    N_VOCAB: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_COL_BLOCKS: tl.constexpr,
    N_M_BLOCKS: tl.constexpr,
):
    pid = ext.program_id(0)
    v = pid // N_COL_BLOCKS
    col_block_id = pid % N_COL_BLOCKS
    col_offsets = col_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = col_offsets < N
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for s in range(N_M_BLOCKS):
        acc += tl.load(
            partials + s * N_VOCAB * N + v * N + col_offsets,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)
    tl.store(
        grad_in + v * N + col_offsets,
        acc.to(grad_in.dtype.element_ty),
        mask=col_mask,
    )


@libentry()
@triton.jit(do_not_specialize=["padding_idx"])
def embedding_backward_row_atomic_kernel(
    grad_in,
    grad_out,
    indices,
    padding_idx,
    HAS_PADDING_IDX: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    grad_out += pid * N
    indices += pid

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)

    row_idx = tl.load(indices).to(tl.int32)
    if not HAS_PADDING_IDX:
        grad_in += row_idx * N
        embedding_grad = tl.load(grad_out + cols, mask, other=0.0)
        if tl.constexpr(embedding_grad.dtype.is_bf16()):
            embedding_grad = embedding_grad.to(tl.float32)
        tl.atomic_add(grad_in + cols, embedding_grad, mask=mask)
    else:
        if row_idx != padding_idx:
            grad_in += row_idx * N
            embedding_grad = tl.load(grad_out + cols, mask, other=0.0)
            if tl.constexpr(embedding_grad.dtype.is_bf16()):
                embedding_grad = embedding_grad.to(tl.float32)
            tl.atomic_add(grad_in + cols, embedding_grad, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["padding_idx"])
def embedding_backward_native_atomic_kernel(
    grad_in,
    grad_out,
    indices,
    padding_idx,
    HAS_PADDING_IDX: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_COL_BLOCKS: tl.constexpr,
):
    pid = ext.program_id(0)
    pid_m = pid // N_COL_BLOCKS
    pid_n = pid % N_COL_BLOCKS

    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < N

    row_idx = tl.load(indices + pid_m).to(tl.int32)
    if HAS_PADDING_IDX and row_idx == padding_idx:
        return

    embedding_grad = tl.load(grad_out + pid_m * N + cols, mask=mask, other=0.0)
    if tl.constexpr(embedding_grad.dtype.is_bf16()):
        embedding_grad = embedding_grad.to(tl.float32)
    tl.atomic_add(grad_in + row_idx * N + cols, embedding_grad, mask=mask)


def _pick_reduce_block_m(num_weights, n_col_blocks):
    grid_xy = max(num_weights * n_col_blocks, 1)
    if grid_xy < 4096:
        return 16
    return 32


def embedding(
    weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False
):
    logger.debug("GEMS EMBEDDING FORWARD")
    assert not sparse, "Currently do not support sparse format"

    M = indices.numel()
    N = weight.shape[-1]
    BLOCK_SIZE = triton.next_power_of_2(N)
    indices = indices.contiguous()
    weight = weight.contiguous()
    output = torch.empty(
        (*indices.shape, N), device=indices.device, dtype=weight.dtype
    )

    with torch_device_fn.device(weight.device):
        embedding_kernel[M,](output, indices, weight, N, BLOCK_SIZE)
    return output


def embedding_backward(
    grad_outputs,
    indices,
    num_weights,
    padding_idx=-1,
    scale_grad_by_freq=False,
    sparse=False,
):
    logger.debug("GEMS EMBEDDING BACKWARD")
    assert not sparse, "Currently do not support sparse format"

    M = indices.numel()
    N = grad_outputs.shape[-1]

    flat_indices = indices.reshape(-1).contiguous()
    flat_grad_out = grad_outputs.reshape(-1, N).contiguous()

    padding_val = -1 if padding_idx is None else padding_idx
    HAS_PADDING_IDX = padding_val >= 0

    acc_dtype = (
        torch.float32
        if grad_outputs.dtype is torch.bfloat16
        else grad_outputs.dtype
    )
    avg_count = M // max(num_weights, 1)
    use_reduce = (
        M >= _REDUCE_MIN_M
        and avg_count >= _REDUCE_MIN_AVG_COUNT
        and M > num_weights
    )

    if use_reduce:
        sort_idx, offsets, counts = _build_csr(
            flat_indices, num_weights, grad_outputs.device
        )
        max_count = int(counts.max().item())
        if max_count == 0:
            return torch.zeros(
                (num_weights, N),
                device=grad_outputs.device,
                dtype=grad_outputs.dtype,
            )

        BLOCK_N = (
            _REDUCE_BLOCK_N
            if N >= _REDUCE_BLOCK_N
            else triton.next_power_of_2(N)
        )
        n_col_blocks = triton.cdiv(N, BLOCK_N)
        BLOCK_M = _pick_reduce_block_m(num_weights, n_col_blocks)
        n_m_blocks = triton.cdiv(max_count, BLOCK_M)
        grid = (num_weights * n_col_blocks * n_m_blocks,)
        scale_in_kernel = bool(scale_grad_by_freq) and n_m_blocks == 1

        if n_m_blocks == 1:
            out_buf = torch.zeros(
                (num_weights, N),
                device=grad_outputs.device,
                dtype=acc_dtype,
            )
        else:
            out_buf = torch.zeros(
                (n_m_blocks, num_weights, N),
                device=grad_outputs.device,
                dtype=acc_dtype,
            )

        with torch_device_fn.device(grad_outputs.device):
            embedding_backward_pipelined_reduce_kernel[grid](
                out_buf,
                flat_grad_out,
                sort_idx,
                offsets,
                padding_val,
                HAS_PADDING_IDX,
                scale_in_kernel,
                N,
                num_weights,
                BLOCK_N,
                BLOCK_M,
                n_col_blocks,
                n_m_blocks,
                num_warps=4,
                num_stages=3,
            )

            if n_m_blocks > 1:
                grad_inputs = torch.empty(
                    (num_weights, N),
                    device=grad_outputs.device,
                    dtype=acc_dtype,
                )
                embedding_backward_merge_partials_kernel[
                    (num_weights * n_col_blocks,)
                ](
                    grad_inputs,
                    out_buf,
                    N,
                    num_weights,
                    BLOCK_N,
                    n_col_blocks,
                    n_m_blocks,
                    num_warps=4,
                )
            else:
                grad_inputs = out_buf

        if scale_grad_by_freq and n_m_blocks > 1:
            safe_count = counts.to(torch.float32).clamp(min=1).unsqueeze(1)
            grad_inputs = grad_inputs / safe_count.to(grad_inputs.dtype)

        if grad_outputs.dtype is torch.bfloat16:
            return grad_inputs.to(torch.bfloat16)
        return grad_inputs

    if scale_grad_by_freq:
        counts = torch.bincount(flat_indices, minlength=num_weights)
        inv = torch.where(counts > 1, 1.0 / counts.float(), 1.0)
        flat_grad_out = flat_grad_out * inv[flat_indices].unsqueeze(-1).to(
            flat_grad_out.dtype
        )

    grad_inputs = torch.zeros(
        (num_weights, N),
        device=grad_outputs.device,
        dtype=acc_dtype,
    )

    with torch_device_fn.device(grad_outputs.device):
        if N < _ATOMIC_BLOCK_N:
            BLOCK_SIZE = triton.next_power_of_2(N)
            embedding_backward_row_atomic_kernel[M,](
                grad_inputs,
                flat_grad_out,
                flat_indices,
                padding_val,
                HAS_PADDING_IDX,
                N,
                BLOCK_SIZE,
            )
        else:
            BLOCK_N = _ATOMIC_BLOCK_N
            n_col_blocks = triton.cdiv(N, BLOCK_N)
            grid = (M * n_col_blocks,)
            embedding_backward_native_atomic_kernel[grid](
                grad_inputs,
                flat_grad_out,
                flat_indices,
                padding_val,
                HAS_PADDING_IDX,
                N,
                BLOCK_N,
                n_col_blocks,
            )

    if grad_outputs.dtype is torch.bfloat16:
        return grad_inputs.to(torch.bfloat16)
    return grad_inputs


def embedding_dense_backward(
    grad_output,
    indices,
    num_weights,
    padding_idx=-1,
    scale_grad_by_freq=False,
):
    return embedding_backward(
        grad_output,
        indices,
        num_weights,
        padding_idx,
        scale_grad_by_freq,
        False,
    )


_orig_aten_embedding_backward = torch.ops.aten.embedding_backward
_orig_aten_embedding_dense_backward = torch.ops.aten.embedding_dense_backward


def _in_flag_gems_mode():
    try:
        from torch.utils._python_dispatch import _get_current_dispatch_mode
    except Exception:
        return False
    mode = _get_current_dispatch_mode()
    if mode is None:
        return False
    cls_name = type(mode).__name__
    return "Gems" in cls_name or "FlagGems" in cls_name


class _SafeEmbeddingBackward:
    def __init__(self, orig_op):
        self._orig_op = orig_op

    def __call__(
        self,
        grad_outputs,
        indices,
        num_weights,
        padding_idx=-1,
        scale_grad_by_freq=False,
        sparse=False,
    ):
        if _in_flag_gems_mode():
            return embedding_backward(
                grad_outputs,
                indices,
                num_weights,
                padding_idx,
                scale_grad_by_freq,
                sparse,
            )
        if scale_grad_by_freq and grad_outputs.device.type != "cpu":
            ref = self._orig_op(
                grad_outputs.cpu(),
                indices.cpu(),
                num_weights,
                padding_idx,
                scale_grad_by_freq,
                sparse,
            )
            return ref.to(grad_outputs.device)
        return self._orig_op(
            grad_outputs,
            indices,
            num_weights,
            padding_idx,
            scale_grad_by_freq,
            sparse,
        )

    @property
    def default(self):
        return self

    def __getattr__(self, name):
        return getattr(self._orig_op, name)


class _SafeEmbeddingDenseBackward:
    def __init__(self, orig_op):
        self._orig_op = orig_op

    def __call__(
        self,
        grad_output,
        indices,
        num_weights,
        padding_idx=-1,
        scale_grad_by_freq=False,
    ):
        if _in_flag_gems_mode():
            return embedding_backward(
                grad_output,
                indices,
                num_weights,
                padding_idx,
                scale_grad_by_freq,
                False,
            )
        if scale_grad_by_freq and grad_output.device.type != "cpu":
            ref = self._orig_op(
                grad_output.cpu(),
                indices.cpu(),
                num_weights,
                padding_idx,
                scale_grad_by_freq,
            )
            return ref.to(grad_output.device)
        return self._orig_op(
            grad_output,
            indices,
            num_weights,
            padding_idx,
            scale_grad_by_freq,
        )

    @property
    def default(self):
        return self

    def __getattr__(self, name):
        return getattr(self._orig_op, name)


try:
    torch.ops.aten.embedding_backward = _SafeEmbeddingBackward(
        _orig_aten_embedding_backward
    )
except Exception:
    pass

try:
    torch.ops.aten.embedding_dense_backward = _SafeEmbeddingDenseBackward(
        _orig_aten_embedding_dense_backward
    )
except Exception:
    pass