import infini.ops
import pytest
import torch

from tests.utils import Payload, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shapes, dim",
    (
        (((4, 3), (4, 5)), 1),
        (((2, 3), (4, 3)), 0),
        (((2, 3, 4), (2, 5, 4)), 1),
        (((2, 3, 4), (2, 3, 6)), 2),
        (((2, 3, 4), (2, 3, 4), (2, 3, 4)), 0),
        (((1, 8), (3, 8), (2, 8)), 0),
        (((3, 1), (3, 2), (3, 4)), 1),
        (((2, 3, 4), (2, 3, 4)), -1),
        (((2, 3, 4), (2, 3, 4)), -2),
        (((16, 128), (16, 256)), 1),
    ),
)
def test_cat(shapes, dim, dtype, device, rtol, atol):
    inputs = [
        randn_strided(shape, None, dtype=dtype, device=device)
        for shape in shapes
    ]

    expected_shape = list(shapes[0])
    cat_dim = dim if dim >= 0 else dim + len(shapes[0])
    expected_shape[cat_dim] = sum(s[cat_dim] for s in shapes)

    out = torch.empty(expected_shape, dtype=dtype, device=device)

    return Payload(
        _cat, _torch_cat, (inputs, dim, out), {}, rtol=rtol, atol=atol
    )


def _cat(inputs, dim, out):
    infini.ops.cat(inputs[0], inputs[1:], dim, out)

    return out


def _torch_cat(inputs, dim, out):
    result = torch.cat(inputs, dim=dim)
    out.copy_(result)

    return out
