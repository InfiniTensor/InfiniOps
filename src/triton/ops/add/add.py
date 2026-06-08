import triton
import triton.language as tl


@triton.jit
def kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    out_shape_ptr,
    x_stride_ptr,
    y_stride_ptr,
    out_stride_ptr,
    x_contig,
    y_contig,
    out_contig,
    ndim,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
    mask = offsets < n_elements

    if (x_contig != 0) and (y_contig != 0) and (out_contig != 0):
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)
    else:
        x_offs = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
        y_offs = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
        out_offs = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
        tmp = offsets

        for i in range(ndim):
            s = tl.load(out_shape_ptr + (ndim - 1 - i))
            d = tmp % s
            tmp = tmp // s
            x_offs += d * tl.load(x_stride_ptr + (ndim - 1 - i))
            y_offs += d * tl.load(y_stride_ptr + (ndim - 1 - i))
            out_offs += d * tl.load(out_stride_ptr + (ndim - 1 - i))

        if x_contig != 0:
            x_offs = offsets
        if y_contig != 0:
            y_offs = offsets
        if out_contig != 0:
            out_offs = offsets

        x = tl.load(x_ptr + x_offs, mask=mask)
        y = tl.load(y_ptr + y_offs, mask=mask)
        tl.store(out_ptr + out_offs, x + y, mask=mask)
