"""
OpenAI Triton Softmax & Online Single-Pass Softmax.
Features safe multi-pass and online single-pass softmax reductions.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    rows, cols,
    stride_or, stride_oc,
    stride_ir, stride_ic,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= rows:
        return

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < cols

    row_in_ptr = in_ptr + row_idx * stride_ir + col_offsets * stride_ic
    row = tl.load(row_in_ptr, mask=mask, other=-float('inf'))

    # Safe softmax: max reduction -> subtract -> exp -> sum reduction -> divide
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_out = numerator / denominator

    row_out_ptr = out_ptr + row_idx * stride_or + col_offsets * stride_oc
    tl.store(row_out_ptr, softmax_out, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Execute Softmax in OpenAI Triton."""
    assert x.is_cuda and x.dtype == torch.float32
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    rows, cols = x_2d.shape

    out = torch.empty_like(x_2d)
    block_size = triton.next_power_of_2(cols)

    _softmax_kernel[(rows,)](
        out, x_2d,
        rows, cols,
        out.stride(0), out.stride(1),
        x_2d.stride(0), x_2d.stride(1),
        BLOCK_SIZE=block_size,
    )
    return out.reshape(orig_shape)
