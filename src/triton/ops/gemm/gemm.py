import triton
import triton.language as tl


@triton.jit
def kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha,
    beta,
    m,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    batch_stride_a,
    batch_stride_b,
    batch_stride_c,
    batch_count,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = tl.program_id(1)

    num_pid_m = tl.cdiv(m, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(n, BLOCK_SIZE_N)

    group_size = GROUP_SIZE_M * num_pid_n
    group = pid // group_size
    first_pid_m = group * GROUP_SIZE_M
    group_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + (pid % group_m)
    pid_n = (pid % group_size) // group_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_base = a_ptr + batch * batch_stride_a
    b_base = b_ptr + batch * batch_stride_b
    c_base = c_ptr + batch * batch_stride_c

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in tl.range(0, k, BLOCK_SIZE_K):
        k_idxs = k_start + offs_k

        a = tl.load(
            a_base + offs_m[:, None] * stride_am + k_idxs[None, :] * stride_ak,
            mask=(offs_m[:, None] < m) & (k_idxs[None, :] < k),
            other=0.0,
        )
        b = tl.load(
            b_base + k_idxs[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k_idxs[:, None] < k) & (offs_n[None, :] < n),
            other=0.0,
        )

        acc = tl.dot(a, b, acc)

    c_offsets = c_base + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)

    c = tl.load(c_offsets, mask=c_mask, other=0.0)
    out = alpha * acc + beta * c

    tl.store(c_offsets, out, mask=c_mask)
