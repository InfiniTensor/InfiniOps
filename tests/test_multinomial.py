import infini.ops
import pytest
import torch

from tests.utils import get_stream


@pytest.mark.parametrize("use_deprecated", (False, True))
@pytest.mark.parametrize("dtype", (torch.float32,))
def test_multinomial(use_deprecated, implementation_index, dtype, device):
    input = torch.tensor(
        ((0.1, 0.2, 0.3, 0.4), (0.4, 0.3, 0.2, 0.1)),
        dtype=dtype,
        device=device,
    )
    out = torch.empty((2, 2), dtype=torch.int64, device=device)
    seed = 20260729
    kwargs = {
        "implementation_index": implementation_index,
        "stream": get_stream(device),
    }

    if use_deprecated:
        torch.manual_seed(seed)
        expected = torch.multinomial(input, 2, replacement=False)
        torch.manual_seed(seed)
        infini.ops.multinomial(input, 2, False, out, **kwargs)
    else:
        try:
            generator = torch.Generator(device=device).manual_seed(seed)
            expected_generator = torch.Generator(device=device).manual_seed(seed)
        except RuntimeError as error:
            pytest.skip(f"device generator is unavailable: {error}")

        infini.ops.multinomial(input, 2, False, generator, out, **kwargs)
        expected = torch.multinomial(
            input, 2, replacement=False, generator=expected_generator
        )

    assert torch.equal(out, expected)
