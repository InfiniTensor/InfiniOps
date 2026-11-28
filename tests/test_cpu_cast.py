import infini.ops
import pytest
import torch


@pytest.fixture(autouse=True)
def require_native_cpu_cast():
    if 0 not in infini.ops.Cast.active_implementation_indices("cpu"):
        pytest.skip("native CPU Cast is not active")

    infini.ops.Cast.clear_cache()


def cast(source, out):
    infini.ops.cast(source, out, implementation_index=0)


DTYPES = [
    torch.float32,
    torch.float64,
    torch.float16,
    torch.bfloat16,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
]


@pytest.mark.smoke
@pytest.mark.parametrize("source_dtype", DTYPES)
@pytest.mark.parametrize("out_dtype", DTYPES)
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "gapped"])
def test_cpu_cast_conversions(source_dtype, out_dtype, layout):
    source = torch.arange(6, dtype=source_dtype).reshape(2, 3)
    backing = torch.full((2, 3, 2), 17, dtype=out_dtype)
    out = backing[..., 0]

    if layout == "contiguous":
        out = torch.empty((2, 3), dtype=out_dtype)
    elif layout == "transpose":
        source = source.t()
        out = out.t()
    else:
        source_backing = torch.full((2, 3, 2), 17, dtype=source_dtype)
        source_backing[..., 0] = source
        source = source_backing[..., 0]

    expected = source.to(out_dtype).clone()
    cast(source, out)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert torch.all(backing[..., 1] == 17)


@pytest.mark.smoke
@pytest.mark.parametrize("shape", [(), (0, 3), (2, 0, 3), (2, 3)])
def test_cpu_cast_scalar_empty_and_same_view(shape):
    source = torch.ones(shape, dtype=torch.float32)
    out = torch.empty(shape, dtype=torch.float16)
    cast(source, out)
    cast(out, out)
    torch.testing.assert_close(out, source.half(), rtol=0, atol=0)


def test_cpu_cast_broadcast_input():
    source = torch.tensor([1.0, 2.0, 3.0]).expand(4, 3)
    out = torch.empty((4, 3), dtype=torch.float16)
    cast(source, out)
    torch.testing.assert_close(out, source.half(), rtol=0, atol=0)
    cast(source, source)
    torch.testing.assert_close(source[0], torch.tensor([1.0, 2.0, 3.0]))


@pytest.mark.smoke
@pytest.mark.parametrize("offset", [-1, 0, 1])
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.float16])
def test_cpu_cast_rejects_overlap_after_cached_call(offset, out_dtype):
    backing = torch.arange(12, dtype=torch.float32)
    source = backing[2:6]
    target = backing[2 + offset : 6 + offset]

    if out_dtype == torch.float16:
        target = target.view(torch.float16)[:4]
    elif offset == 0:
        # A different traversal of the same storage is not an identity copy.
        source = source.view(2, 2)
        target = target.view(2, 2).t()

    cast(source, torch.empty_strided(target.shape, target.stride(), dtype=out_dtype))
    before = backing.clone()

    with pytest.raises(ValueError):
        cast(source, target)

    torch.testing.assert_close(backing, before, rtol=0, atol=0)


@pytest.mark.parametrize("strides", [(0, 1), (1, 1)])
def test_cpu_cast_rejects_overlapping_output(strides):
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    backing = torch.full((6,), 17.0, dtype=torch.float16)
    target = backing.as_strided((2, 3), strides)

    with pytest.raises(ValueError):
        cast(source, target)

    assert torch.all(backing == 17)


def test_cpu_cast_rejects_shape_change():
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    out = torch.full((3, 2), 17.0, dtype=torch.float16)

    with pytest.raises(ValueError):
        cast(source, out)

    assert torch.all(out == 17)
