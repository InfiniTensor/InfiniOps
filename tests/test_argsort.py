import infini.ops
import pytest

import torch
from tests.utils import get_stream


@pytest.mark.parametrize(
    ("overload", "dim", "descending", "stable"),
    (
        ("default", -1, False, False),
        ("dim", 0, False, False),
        ("descending", -1, True, False),
        ("stable", -1, False, True),
        ("deprecated", 0, True, True),
    ),
)
@pytest.mark.parametrize("dtype", (torch.float32,))
def test_argsort(
    overload,
    dim,
    descending,
    stable,
    implementation_index,
    dtype,
    device,
):
    input = torch.tensor(
        ((2.0, 1.0, 1.0), (0.0, 3.0, -1.0)), dtype=dtype, device=device
    )
    out = torch.empty_like(input, dtype=torch.int64)
    kwargs = {
        "stream": get_stream(device),
        "implementation_index": implementation_index,
    }

    if overload == "default":
        infini.ops.argsort(input, out, **kwargs)
    elif overload == "dim":
        infini.ops.argsort(input, dim, out, **kwargs)
    elif overload == "descending":
        infini.ops.argsort(input, dim, descending, out, **kwargs)
    elif overload == "stable":
        infini.ops.argsort(input, dim, descending, stable, out, **kwargs)
    else:
        infini.ops.argsort(input, stable, dim, descending, out, **kwargs)

    expected = torch.argsort(input, dim=dim, descending=descending, stable=stable)
    assert torch.equal(out, expected)

    if overload == "deprecated":
        new_order_conversion = torch.argsort(
            input,
            dim=int(stable),
            descending=bool(dim),
            stable=descending,
        )
        assert not torch.equal(out, new_order_conversion)
