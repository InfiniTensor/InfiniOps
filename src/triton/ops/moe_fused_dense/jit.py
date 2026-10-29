import triton
import triton.language as tl


@triton.jit
def moe_fused_dense_kernel(
    X,
    W13,
    W2,
    Weights,
    Ids,
    Sorted,
    Experts,
    Total,
    Mid,
    Pairs,
    Out,
    TOKENS: tl.constexpr,
    E: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,  # noqa: E741
    TOPK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    STAGE: tl.constexpr,
):
    if STAGE == 2:
        token = tl.program_id(0)
        h = tl.program_id(1) * BN + tl.arange(0, BN)
        ranks = tl.arange(0, triton.next_power_of_2(TOPK))
        route = tl.load(Ids + token * TOPK + ranks, ranks < TOPK, E)
        active = (route >= 0) & (route < E)
        values = tl.load(
            Pairs + (token * TOPK + ranks[:, None]) * H + h[None, :],
            active[:, None] & (h[None, :] < H),
            0,
        ).to(tl.float32)
        weights = tl.load(Weights + token * TOPK + ranks, ranks < TOPK, 0)
        tl.store(Out + token * H + h, tl.sum(values * weights[:, None], 0), h < H)
    else:
        block = tl.program_id(0)
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        if block * BM < tl.load(Total):
            expert = tl.load(Experts + block)
            rows = tl.load(Sorted + block * BM + tl.arange(0, BM))
            valid = rows < TOKENS * TOPK
            kk = tl.arange(0, BK)
            acc = tl.zeros((BM, BN), tl.float32)
            if expert >= 0:
                if STAGE == 0:
                    for start in range(tl.cdiv(H, BK)):
                        k = start * BK + kk
                        x = tl.load(
                            X + (rows[:, None] // TOPK) * H + k[None, :],
                            valid[:, None] & (k[None, :] < H),
                            0,
                        )
                        w = tl.load(
                            W13 + expert * (2 * I * H) + n[None, :] * H + k[:, None],
                            (n[None, :] < 2 * I) & (k[:, None] < H),
                            0,
                        )
                        acc += tl.dot(x, w)
                    tl.store(
                        Mid + rows[:, None] * (2 * I) + n[None, :],
                        acc,
                        valid[:, None] & (n[None, :] < 2 * I),
                    )
                else:
                    for start in range(tl.cdiv(I, BK)):
                        k = start * BK + kk
                        gate = tl.load(
                            Mid + rows[:, None] * (2 * I) + k[None, :],
                            valid[:, None] & (k[None, :] < I),
                            0,
                        ).to(tl.float32)
                        up = tl.load(
                            Mid + rows[:, None] * (2 * I) + I + k[None, :],
                            valid[:, None] & (k[None, :] < I),
                            0,
                        ).to(tl.float32)
                        x = (gate * tl.sigmoid(gate) * up).to(Mid.dtype.element_ty)
                        w = tl.load(
                            W2 + expert * (H * I) + n[None, :] * I + k[:, None],
                            (n[None, :] < H) & (k[:, None] < I),
                            0,
                        )
                        acc += tl.dot(x, w)
                    tl.store(
                        Pairs + rows[:, None] * H + n[None, :],
                        acc,
                        valid[:, None] & (n[None, :] < H),
                    )
            elif STAGE == 1:
                tl.store(
                    Pairs + rows[:, None] * H + n[None, :],
                    acc,
                    valid[:, None] & (n[None, :] < H),
                )


kernel = moe_fused_dense_kernel
