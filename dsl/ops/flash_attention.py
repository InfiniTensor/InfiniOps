from dsl.decorators import manual_op


@manual_op(
    name="FlashAttention",
    base="src/base/flash_attention.h",
    backends={
        "ascend": "ascend/flash_attention/kernel.h",
    },
)
def flash_attention():
    ...
