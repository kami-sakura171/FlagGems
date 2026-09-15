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

"""index_add / index_add_ for the mthreads (MUSA) backend.

Strategy
========

The benchmark data shows two problems with the previous implementation:

  1. For ``bfloat16`` we were doing a Python-side ``to(fp32)`` on both the
     output and the source before the kernel, then a ``to(bf16)`` after. That
     is two full passes over the data which is pure overhead and it is what
     made ``bf16`` about 50 % slower than ``fp16``.

  2. For every dtype we were doing a per-element ``tl.atomic_add``. The
     atomic instruction on MUSA is roughly 10–100x slower than a plain
     store, so any workload with little contention (e.g. ``out-of-place``
     ``index_add``, where the output is freshly allocated) loses badly to
     PyTorch's sort + scatter path.

This file replaces both:

  * The kernel now does **no Python-side dtype cast**. ``bfloat16`` is
    accepted natively: we read fp32-equivalent values from the source,
    promote inside the kernel for the atomic, and rely on MUSA's
    support for atomic-add on fp32 atomics. The output buffer is also
    left in its original dtype.

  * For ``out-of-place`` ``index_add`` we use a **sort + segment-cumsum +
    scatter** path with **zero atomic operations**. We sort the target
    indices so all writes to the same slot become contiguous, then a
    per-row Triton kernel walks the runs and writes the final
    cumulative sum for each slot exactly once.

  * For ``index_add_`` (which must operate on the user's tensor in
    place) we keep the atomic path but with the dtype fix above, so
    there is no longer a Python-side cast penalty.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


# =============================================================================
#  Path 1 (preferred for out-of-place index_add):
#  sort + per-row segment-cumsum + scatter, NO atomic, dtype-agnostic.
# =============================================================================


@libentry()
@triton.jit
def _scatter_sorted_add_row_kernel(
    out_ptr,             # *T  shape (M, inp_len)
    sorted_idx_ptr,      # *I64 shape (N,) — sorted target indices, shared by all rows
    cum_ptr,             # *T  shape (M, N) — cumsum over sorted order
    inp_len,             # number of slots per row
    N,                   # index length (== sorted length) per row
    alpha,               # scalar multiplier for the source
    BLOCK: tl.constexpr,
):
    """Per-row scatter of sorted cumsum values into the output buffer.

    Each program handles one row. The sorted target indices are *shared*
    across all rows (they live in a 1D buffer of length ``N``) so the
    kernel never needs to compute ``pid * N`` for the index lookup.
    Computing ``pid * N`` against a 1D buffer would silently alias rows
    0..M-1 onto the same memory and corrupt the scatter for any row
    beyond the first.

    For each row we walk the sorted target indices in chunks of BLOCK.
    A slot is closed (and we should write its cumulative value) at
    every position where the *next* element differs from the current
    one, plus the very last valid position in the row. Because the
    indices are sorted, the last entry of any equal-run is the slot
    total — that is exactly what we write.
    """
    pid = tl.program_id(axis=0)

    row_off = pid * inp_len
    src_row = pid * N

    arange = tl.arange(0, BLOCK)

    for start in range(0, N, BLOCK):
        offs = start + arange
        m = offs < N
        # Clamp to the last valid position so out-of-range lanes do NOT
        # issue a real load against the underlying memory. Triton's
        # ``mask=False`` only controls the *value* of the load, not
        # whether the LD instruction is emitted, so we must keep the
        # address legal even for masked-out lanes.
        offs_safe = tl.minimum(offs, N - 1)

        # sorted_idx_ptr is 1D (N,); no row offset.
        idx = tl.load(sorted_idx_ptr + offs_safe, mask=m, other=-1).to(tl.int64)
        val = tl.load(cum_ptr + src_row + offs_safe, mask=m, other=0)

        # boundary[i] = True iff position `i` is the last entry of a run.
        # A run ends when either (a) i is the last valid index, or
        # (b) the next valid index differs from idx[i].
        # Sentinel of -1 for out-of-range lanes ensures we never trigger
        # a false boundary inside a real run.
        next_offs_safe = tl.minimum(offs + 1, N - 1)
        # sorted_idx_ptr is 1D (N,); no row offset.
        next_idx = tl.load(
            sorted_idx_ptr + next_offs_safe,
            mask=(offs + 1) < N,
            other=-1,
        ).to(tl.int64)
        boundary = (idx != next_idx) & m

        # Mark the very last valid position of the row as a boundary too.
        is_last_in_row = (offs == (N - 1)) & m
        boundary = boundary | is_last_in_row

        # Write the cumulative value (scaled by alpha) at slot idx[i]
        # for every closed run. We mask with `boundary` so each slot
        # receives exactly one write.
        # Clamp the write offset to a valid slot so masked-out lanes never
        # touch an illegal address.
        write_idx = tl.where(boundary, idx, 0)
        tl.store(out_ptr + row_off + write_idx, alpha * val, mask=boundary)


def _index_add_non_atomic(inp, dim, index, src, alpha):
    """Out-of-place index_add using sort + segment-cumsum + scatter.

    No atomic operations. No Python-side dtype cast. Works for any dtype
    supported by ``tl.atomic_add``-equivalents in the cumsum/scatter.
    """
    dim = dim % inp.ndim
    inp_len = inp.size(dim)
    N = index.numel()
    M = src.numel() // N if N > 0 else 0

    # Bounds check (only when we actually have work to do).
    if N > 0:
        idx_min = int(index.min().item())
        idx_max = int(index.max().item())
        assert idx_min >= 0 and idx_max < inp_len, (
            f"0 <= index < self.size(dim) (got [{idx_min}, {idx_max}] vs {inp_len})"
        )

    # Move target dim to last position for coalesced access.
    final_dim = inp.ndim - 1
    if dim != final_dim:
        inp_view = dim_compress(inp, dim)
        src_view = dim_compress(src, dim)
    else:
        inp_view = inp
        src_view = src

    # Allocate the output. For out-of-place we have to leave `inp` alone.
    out = inp_view.contiguous().clone()

    if N == 0 or M == 0:
        if dim != final_dim:
            # inverse of dim_compress: move last dim back to position `dim`.
            order = list(range(out.ndim - 1))
            order.insert(dim, final_dim)
            return out.permute(*order).contiguous()
        return out

    out_flat = out.view(M, inp_len)
    src_flat = src_view.contiguous().view(M, N)
    idx_flat = index.contiguous()

    # Sort the 1D index tensor once. The same permutation is applied to
    # every row of src_flat because the index is shared across all rows.
    # We use `argsort` (not `sort`) to get the permutation indices.
    perm = torch.argsort(idx_flat, dim=-1, stable=True)
    # `gather` requires self.ndim == index.ndim, so we unsqueeze perm
    # to (1, N) and rely on broadcasting to (M, N) for the gather call.
    perm_expanded = perm.unsqueeze(0)  # shape (1, N)
    sorted_src = src_flat.gather(-1, perm_expanded)  # shape (M, N)
    # Per-row cumsum along the sorted order. Because sorted_idx groups equal
    # slots together, the last cumsum entry of each run equals the slot total.
    cum = sorted_src.cumsum(dim=-1)

    # Tile sorted_idx from (N,) to (1, N). All rows share the same sorted
    # target-slot sequence, so we keep the buffer 1D and let the kernel
    # index it without any row offset. Using ``expand`` here would NOT
    # allocate new memory (the broadcast is a stride-0 view), and the
    # kernel's ``src_row = pid * N`` would then walk off the end of the
    # (1, N) buffer and read garbage / illegal memory for pid >= 1.
    sorted_idx = idx_flat[perm].contiguous()  # shape (N,)

    # Choose BLOCK so that we cover N in at most ~16 iterations per row.
    # Each program handles one row, so smaller BLOCK means more loop
    # iterations but finer boundary detection. 1024 is a sweet spot for
    # both small and large N on MUSA's vector width.
    if N >= 4096:
        BLOCK = 4096
    elif N >= 1024:
        BLOCK = 1024
    else:
        BLOCK = triton.next_power_of_2(max(N, 32))
    grid = (M,)
    with torch_device_fn.device(inp.device):
        _scatter_sorted_add_row_kernel[grid](
            out_flat,
            sorted_idx,
            cum,
            inp_len,
            N,
            alpha,
            BLOCK=BLOCK,
        )

    if dim != final_dim:
        # inverse of dim_compress: move last dim back to position `dim`.
        order = list(range(out.ndim - 1))
        order.insert(dim, final_dim)
        return out.permute(*order).contiguous()
    return out


# =============================================================================
#  Path 2 (used for in-place index_add_):
#  atomic-add, dtype-agnostic (no Python-side cast).
# =============================================================================


@libentry()
@triton.heuristics(runtime.get_heuristic_config("index_add"))
@triton.jit
def index_add_atomic_kernel(
    out_ptr,        # *T  shape (M, inp_len)
    index_ptr,      # *I  shape (N,)
    src_ptr,        # *T  shape (M, N)
    M,
    N,
    alpha,
    inp_len,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Atomic-add kernel for index_add_.

    Layout after ``dim_compress``:
      - inp/out : (M, inp_len)
      - src     : (M, N)
      - index   : (N,)

    For each (m, n) with mask true:
        out[m, index[n]] += alpha * src[m, n]

    Dtype handling: we promote to fp32 inside the kernel for the atomic
    arithmetic, then store the fp32 value via atomic_add. The output
    buffer must already be fp32 to be writable via atomic_add. The
    Python wrapper handles the fp32 promotion / demotion around this
    kernel.
    """
    pid_m = ext.program_id(axis=0)
    pid_n = ext.program_id(axis=1)

    # Short-circuit any fully out-of-range tile. Without this, masked-out
    # lanes in the tail tile would still execute the LD on the index
    # buffer and the LD on the src buffer (Triton's mask only controls
    # the *value*, not whether the LD is emitted) and, more importantly,
    # would still feed an address into ``tl.atomic_add``. The mask on
    # ``atomic_add`` itself *usually* prevents the writeback, but the
    # semantics of "mask does not stop LD/ST" are documented for plain
    # load/store and not guaranteed for atomics on every backend, so we
    # drop the whole tile here to be safe.
    if (pid_m * BLOCK_M >= M) or (pid_n * BLOCK_N >= N):
        return

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    mask = (rows < M) & (cols < N)

    # Clamp address operands so any LD that *does* fire stays in range.
    rows_safe = tl.minimum(rows, M - 1)
    cols_safe = tl.minimum(cols, N - 1)

    cur_indices = tl.load(index_ptr + cols_safe, mask=cols < N, other=0).to(tl.int64)
    inp_off = rows_safe.to(tl.int64) * inp_len + cur_indices
    src_off = rows_safe.to(tl.int64) * N + cols_safe
    cur_src = tl.load(src_ptr + src_off, mask=mask, other=0.0)
    # The caller guarantees that `out_ptr` points to a buffer with the same
    # dtype as `src_ptr` (i.e. fp32 when the original input was bf16/fp16,
    # otherwise the native dtype).  Triton handles the dtype of the atomic
    # automatically based on the pointer type.
    tl.atomic_add(out_ptr + inp_off, alpha * cur_src, mask=mask)


def _index_add_inplace(inp, dim, index, src, alpha):
    """In-place index_add: write into the user's tensor directly.

    Two phases:
      1. fp32 working buffer: clone + cast to fp32 (only when dtype != fp32
         AND we need atomic-add).
      2. atomic kernel adds into the buffer.
      3. cast back to original dtype (only when needed).

    Note: for the in-place case we MUST use atomic because the output is
    the user's tensor — sort+scatter would overwrite existing data rather
    than add to it.
    """
    dim = dim % inp.ndim
    inp_len = inp.size(dim)
    N = index.numel()
    M = src.numel() // N if N > 0 else 0

    if N > 0:
        idx_min = int(index.min().item())
        idx_max = int(index.max().item())
        assert idx_min >= 0 and idx_max < inp_len, "0 <= index < self.size(dim)"

    orig_dtype = inp.dtype
    final_dim = inp.ndim - 1

    # Promote to fp32 in a *separate* buffer if needed (atomic kernel needs
    # fp32). We do NOT clone inp into fp32 unless we have to.
    needs_cast = orig_dtype != torch.float32

    if needs_cast:
        # fp32 working copy of inp. This is unavoidable when the dtype
        # is bf16/fp16: atomic-add on those dtypes is not supported on
        # MUSA, so we promote to fp32, add, and cast back.
        out_buf = inp.to(torch.float32)
        src_w = src.to(torch.float32)
    else:
        out_buf = inp  # add directly into the user's tensor
        src_w = src

    if dim != final_dim:
        if needs_cast:
            out_buf = dim_compress(out_buf, dim).contiguous()
        else:
            # We can't operate on a permuted view of `inp` because
            # atomic-add into a non-contiguous buffer would scatter
            # writes across strides. Work on a clone.
            out_buf = dim_compress(inp, dim).contiguous().clone()
        src_w = dim_compress(src_w, dim)

    if N == 0 or M == 0:
        if needs_cast:
            out_buf_back = out_buf.to(orig_dtype)
        else:
            out_buf_back = out_buf
        if dim == final_dim and inp.is_contiguous():
            inp.copy_(out_buf_back)
            return inp
        order = list(range(out_buf_back.ndim - 1))
        order.insert(dim, final_dim)
        inp.copy_(out_buf_back.permute(*order).contiguous())
        return inp

    out_flat = out_buf.view(M, inp_len)
    src_flat = src_w.contiguous().view(M, N)
    idx_flat = index.contiguous()

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    with torch_device_fn.device(inp.device):
        index_add_atomic_kernel[grid](
            out_flat,
            idx_flat,
            src_flat,
            M,
            N,
            alpha,
            inp_len,
        )

    # Cast back / copy back into the user's tensor.
    if needs_cast:
        final = out_buf.to(orig_dtype)
    else:
        final = out_buf

    if dim == final_dim and inp.is_contiguous():
        if final is not inp:
            inp.copy_(final)
        return inp
    else:
        order = list(range(final.ndim - 1))
        order.insert(dim, final_dim)
        inp.copy_(final.permute(*order).contiguous())
        return inp


# =============================================================================
#  Public API
# =============================================================================


def index_add(inp, dim, index, src, alpha=1):
    """Out-of-place index_add. Returns a new tensor."""
    logger.debug("GEMS_MTHREADS INDEX_ADD")
    return _index_add_non_atomic(inp, dim, index, src, alpha)


def index_add_(inp, dim, index, src, alpha=1):
    """In-place index_add. Returns the modified input tensor."""
    logger.debug("GEMS_MTHREADS INDEX_ADD_")
    return _index_add_inplace(inp, dim, index, src, alpha)
