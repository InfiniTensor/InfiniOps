import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_npu_stream, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shapes, dim, out_shape",
    (
        # 2 inputs, dim=0
        (((4, 64), (4, 64)), 0, (8, 64)),
        # 2 inputs, dim=1
        (((4, 32), (4, 64)), 1, (4, 96)),
        # 2 inputs, dim=-1 (negative dim)
        (((4, 32), (4, 64)), -1, (4, 96)),
        # 3 inputs, dim=1
        (((4, 16), (4, 32), (4, 16)), 1, (4, 64)),
        # 2 inputs, dim=0, 3D
        (((2, 4, 64), (2, 4, 64)), 0, (4, 4, 64)),
        # 2 inputs, dim=2, 3D
        (((2, 4, 32), (2, 4, 64)), 2, (2, 4, 96)),
        # 4 inputs, dim=1
        (((1, 1024), (1, 1024), (1, 1024), (1, 1024)), 1, (1, 4096)),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-7, 1e-7),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 5e-3),
    ),
)
def test_cat(shapes, dim, out_shape, dtype, device, rtol, atol):
    inputs = [
        randn_strided(s, None, dtype=dtype, device=device) for s in shapes
    ]
    out = empty_strided(out_shape, None, dtype=dtype, device=device)

    return Payload(
        lambda *args: _cat(*args, dim=dim),
        lambda *args: _torch_cat(*args, dim=dim),
        (*inputs, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _cat(*args, dim):
    inputs = list(args[:-1])
    out = args[-1]

    first = inputs[0]
    rest = inputs[1:]

    if first.device.type == "npu":
        infini.ops.cat(first, rest, dim, out, stream=get_npu_stream(first))
    else:
        infini.ops.cat(first, rest, dim, out)

    return out


def _torch_cat(*args, dim):
    inputs = list(args[:-1])
    out = args[-1]

    result = torch.cat(inputs, dim=dim)
    out.copy_(result)

    return out
