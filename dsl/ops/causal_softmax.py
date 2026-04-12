from dsl.decorators import manual_op


@manual_op(
    name="CausalSoftmax",
    base="src/base/causal_softmax.h",
    backends={
        "cuda": "cuda/causal_softmax/kernel.h",
        "ascend": "ascend/causal_softmax/kernel.h",
        "cpu": "cpu/causal_softmax/causal_softmax.h",
    },
)
def causal_softmax():
    ...
