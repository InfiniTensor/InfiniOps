import infini.ops
import pytest
import torch

from tests.utils import get_stream


@pytest.mark.parametrize(
    "overload, deprecated",
    (
        ("tensor_float", False),
        ("tensor_tensor", False),
        ("float_tensor", False),
        ("float_float", False),
        ("tensor_float", True),
        ("tensor_tensor", True),
    ),
)
def test_normal(overload, deprecated, device, implementation_index):
    shape = (4, 7)
    mean_tensor = torch.randn(shape, dtype=torch.float32, device=device)
    std_tensor = torch.rand(shape, dtype=torch.float32, device=device) + 0.1
    mean_scalar = 0.25
    std_scalar = 1.5
    expected = torch.empty(shape, dtype=torch.float32, device=device)
    out = torch.empty_like(expected)
    seed = 1234

    if deprecated:
        torch.manual_seed(seed)
        expected_generator = None
        actual_generator = None
    else:
        try:
            expected_generator = torch.Generator(device=device).manual_seed(seed)
            actual_generator = torch.Generator(device=device).manual_seed(seed)
        except RuntimeError as error:
            pytest.skip(f"device generator is unavailable: {error}")

    if overload == "tensor_float":
        mean, std = mean_tensor, std_scalar
        torch.normal(mean, std, generator=expected_generator, out=expected)
    elif overload == "tensor_tensor":
        mean, std = mean_tensor, std_tensor
        torch.normal(mean, std, generator=expected_generator, out=expected)
    elif overload == "float_tensor":
        mean, std = mean_scalar, std_tensor
        torch.normal(mean, std, generator=expected_generator, out=expected)
    else:
        mean, std = mean_scalar, std_scalar
        torch.normal(
            mean,
            std,
            shape,
            generator=expected_generator,
            out=expected,
        )

    if deprecated:
        torch.manual_seed(seed)
        infini.ops.normal(
            mean,
            std,
            out,
            stream=get_stream(device),
            implementation_index=implementation_index,
        )
    else:
        kwargs = {
            "mean": mean,
            "std": std,
            "generator": actual_generator,
            "out": out,
            "stream": get_stream(device),
            "implementation_index": implementation_index,
        }
        if overload == "float_float":
            kwargs["size"] = shape
        infini.ops.normal(**kwargs)

    torch.testing.assert_close(out, expected)
