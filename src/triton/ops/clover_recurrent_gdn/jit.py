import triton
import triton.language as tl


@triton.jit
def clover_recurrent_gdn_kernel(
    Q,
    K,
    V,
    G,
    Beta,
    Offsets,
    Initial,
    Final,
    State,
    Out,
    qs,
    qh,
    ks,
    kh,
    vs,
    vh,
    gs,
    gh,
    bs,
    bh,
    ss,
    sh,
    sv,
    sk,
    os,
    oh,
    TOKENS: tl.constexpr,
    POOL: tl.constexpr,
    HK: tl.constexpr,
    HV: tl.constexpr,
    BV: tl.constexpr,
):
    request, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    begin, end = tl.load(Offsets + request), tl.load(Offsets + request + 1)
    initial, final = tl.load(Initial + request), tl.load(Final + request)
    if (
        begin >= 0
        and end >= begin
        and end <= TOKENS
        and initial >= 0
        and initial < POOL
        and final >= 0
        and final < POOL
    ):
        key = tl.arange(0, 128)
        value = tile * BV + tl.arange(0, BV)
        state = tl.load(
            State + initial * ss + head * sh + value[:, None] * sv + key[None, :] * sk
        )
        key_head = head // (HV // HK)
        for token in range(begin, end):
            q = tl.load(Q + token * qs + key_head * qh + key).to(tl.float32)
            k = tl.load(K + token * ks + key_head * kh + key).to(tl.float32)
            v = tl.load(V + token * vs + head * vh + value).to(tl.float32)
            q = q * tl.rsqrt(tl.sum(q * q, 0) + 1.0e-6) * 0.08838834764831845
            k = k * tl.rsqrt(tl.sum(k * k, 0) + 1.0e-6)
            decay = tl.exp(tl.load(G + token * gs + head * gh))
            beta = tl.load(Beta + token * bs + head * bh)
            state = state * decay
            remembered = tl.sum(state * k[None, :], 1)
            delta = (v - remembered) * beta
            state += delta[:, None] * k[None, :]
            y = tl.sum(state * q[None, :], 1)
            tl.store(Out + token * os + head * oh + value, y)
        tl.store(
            State + final * ss + head * sh + value[:, None] * sv + key[None, :] * sk,
            state,
        )


kernel = clover_recurrent_gdn_kernel
