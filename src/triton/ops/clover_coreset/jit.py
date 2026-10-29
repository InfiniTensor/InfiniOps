import triton
import triton.language as tl


@triton.jit
def clover_coreset_kernel(
    Logits, Mass, Allow, Out, top_p, WIDTH: tl.constexpr, STAGE: tl.constexpr
):
    expert = tl.arange(0, 256)
    if STAGE == 0:
        block = tl.program_id(0)
        tokens = tl.arange(0, 32)
        logits = tl.load(
            Logits + (block * WIDTH + tokens[:, None]) * 256 + expert[None, :],
            tokens[:, None] < WIDTH,
            -float("inf"),
        ).to(tl.float32)
        row_max = tl.max(logits, 1)
        probability = tl.exp(logits - row_max[:, None])
        probability = probability / tl.sum(probability, 1)[:, None]
        selected = tl.full((32, 256), False, tl.int1)
        remaining = logits
        for rank in tl.static_range(8):
            best = tl.argmax(remaining, 1, tie_break_left=True)
            chosen = expert[None, :] == best[:, None]
            selected |= chosen
            remaining = tl.where(chosen, -float("inf"), remaining)
        mass = tl.sum(
            tl.where(selected & (tokens[:, None] < WIDTH), probability, 0.0), 0
        )
        tl.store(Mass + block * 256 + expert, mass)
    elif STAGE == 1:
        block = tl.program_id(0)
        mass = tl.load(Mass + block * 256 + expert)
        # Positive float bits sort by magnitude; low bits break ties by expert ID.
        bits = mass.to(tl.int32, bitcast=True).to(tl.int64)
        key = ((0x7FFFFFFF - bits) << 32) | expert.to(tl.int64)
        order = (tl.sort(key, descending=False) & 0xFFFFFFFF).to(tl.int32)
        sorted_mass = tl.load(Mass + block * 256 + order)
        probabilities = sorted_mass / tl.maximum(tl.sum(mass, 0), 1.0e-9)
        keep = tl.cumsum(probabilities, 0) - probabilities < top_p
        keep = tl.where(top_p >= 1.0, sorted_mass > 0.0, keep)
        keep = keep | (expert == 0)
        tl.store(Allow + block * 256 + order, keep)
    else:
        token = tl.program_id(0)
        logits = tl.load(Logits + token * 256 + expert)
        keep = tl.load(Allow + (token // WIDTH) * 256 + expert)
        tl.store(Out + token * 256 + expert, tl.where(keep != 0, logits, -float("inf")))


kernel = clover_coreset_kernel
