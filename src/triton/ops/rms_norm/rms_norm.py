import triton
import triton.language as tl


@triton.jit
def kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    eps,
    m,
    n,
    stride_xm,
    stride_xn,
    stride_wn,
    stride_ym,
    stride_yn,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= m:
        return

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    x_row_ptr = x_ptr + pid * stride_xm
    y_row_ptr = y_ptr + pid * stride_ym

    x = tl.load(x_row_ptr + offs * stride_xn, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs * stride_wn, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x * x) / n
    rrms = 1.0 / tl.sqrt(mean_sq + eps)

    y = x * rrms * w

    tl.store(y_row_ptr + offs * stride_yn, y, mask=mask)
