import infini.ops
import pytest
import torch

from tests.utils import get_stream


@pytest.mark.parametrize("api_style", ("canonical", "deprecated"))
def test_bernoulli(api_style, device, implementation_index):
    seed = 0
    input = torch.linspace(0.05, 0.95, 257, device=device)
    out = torch.empty_like(input)

    if api_style == "canonical":
        try:
            expected_generator = torch.Generator(device=device).manual_seed(seed)
            actual_generator = torch.Generator(device=device).manual_seed(seed)
        except RuntimeError as error:
            pytest.skip(f"device generator is unavailable: {error}")

        expected = torch.bernoulli(input, generator=expected_generator)
        args = (input, actual_generator, out)
    else:
        torch.manual_seed(seed)
        expected = torch.bernoulli(input)
        torch.manual_seed(seed)
        args = (input, out)

    infini.ops.bernoulli(
        *args,
        stream=get_stream(input.device),
        implementation_index=implementation_index,
    )
    torch.testing.assert_close(out, expected, rtol=0.0, atol=0.0)
