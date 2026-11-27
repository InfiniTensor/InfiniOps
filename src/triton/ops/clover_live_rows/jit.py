import triton
import triton.language as tl


@triton.jit
def clover_live_rows_kernel(
    A,
    B,
    C,
    D,
    N: tl.constexpr,
    WIDTH: tl.constexpr,
    MASK: tl.constexpr,
    REFRESH: tl.constexpr,
    STAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < N
    if STAGE == 0:
        current = tl.load(A + offset, valid, 0)
        previous = tl.load(B + offset, valid & (not REFRESH), MASK)
        live = (current == MASK) | (previous == MASK) | REFRESH
        tl.store(C + offset, live, valid)
        tl.store(B + offset, current, valid)
    elif STAGE == 1:
        route = tl.load(A + offset, valid, MASK)
        live = tl.load(B + offset // WIDTH, valid, 0)
        tl.store(C + offset, tl.where(live != 0, route, MASK), valid)
    else:
        fresh = tl.load(A + offset, valid, 0)
        live = tl.load(B + offset // WIDTH, valid, 0)
        cached = tl.load(C + offset, valid & (live == 0), 0)
        output = tl.where(live != 0, fresh, cached)
        tl.store(D + offset, output, valid)
        tl.store(C + offset, output, valid)


kernel = clover_live_rows_kernel
