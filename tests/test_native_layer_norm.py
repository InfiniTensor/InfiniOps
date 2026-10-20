import pytest
import torch

import infini.ops

from tests.utils import get_stream


@pytest.mark.parametrize("shape", [(1, 1, 512), (3, 17, 512), (2, 7, 32)])
@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("eps", [1e-5, 1e-12])
def test_encoder_layer_norm_preserves_all_three_outputs(device, shape, affine, eps):
    x = torch.randn(shape, device=device, dtype=torch.float32)
    normalized_shape = [shape[-1]]
    weight = torch.randn(normalized_shape, device=device) if affine else None
    bias = torch.randn(normalized_shape, device=device) if affine else None
    reference = torch.native_layer_norm(x, normalized_shape, weight, bias, eps)
    output_shapes = [shape, (*shape[:-1], 1), (*shape[:-1], 1)]
    storage = [torch.full((*s, 2), -123.0, device=device) for s in output_shapes]
    outputs = [s[..., 0] for s in storage]
    addresses = [out.data_ptr() for out in outputs]
    infini.ops.native_layer_norm(
        x,
        normalized_shape,
        weight,
        bias,
        eps,
        *outputs,
        implementation_index=8,
        stream=get_stream(x.device),
    )

    for actual, expected, backing, address in zip(
        outputs, reference, storage, addresses
    ):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        assert actual.data_ptr() == address
        assert torch.all(backing[..., 1] == -123.0)
