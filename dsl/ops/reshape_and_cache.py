from dsl.decorators import manual_op


@manual_op(
    name="ReshapeAndCache",
    base="src/base/reshape_and_cache.h",
    backends={
        "ascend": "ascend/reshape_and_cache/kernel.h",
    },
)
def reshape_and_cache():
    ...
