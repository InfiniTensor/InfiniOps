from dsl.decorators import manual_op


@manual_op(
    name="RotaryEmbedding",
    base="src/base/rotary_embedding.h",
    backends={
        "ascend": "ascend/rotary_embedding/kernel.h",
    },
)
def rotary_embedding():
    ...
