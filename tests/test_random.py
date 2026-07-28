import infini.ops
import pytest
import torch

from tests.utils import get_stream


@pytest.mark.parametrize(
    ("schema", "use_deprecated"),
    (
        pytest.param("from", False, id="from-generator"),
        pytest.param("to", False, id="to-generator"),
        pytest.param("from", True, id="from-deprecated"),
        pytest.param("to", True, id="to-deprecated"),
    ),
)
def test_random(schema, use_deprecated, implementation_index, device):
    seed = 20260729
    input = torch.empty((32,), dtype=torch.int64, device=device)
    expected = torch.empty_like(input)
    kwargs = {
        "implementation_index": implementation_index,
        "stream": get_stream(device),
    }

    if use_deprecated:
        if schema == "from":
            infini.ops.random(input, 5, 23, **kwargs)
        else:
            infini.ops.random(input, 23, **kwargs)
    else:
        try:
            generator = torch.Generator(device=device).manual_seed(seed)
            expected_generator = torch.Generator(device=device).manual_seed(seed)
        except RuntimeError as error:
            pytest.skip(f"device generator is unavailable: {error}")

        if schema == "from":
            expected.random_(5, 23, generator=expected_generator)
            infini.ops.random(input, 5, 23, generator, **kwargs)
        else:
            expected.random_(23, generator=expected_generator)
            infini.ops.random(input, 23, generator, **kwargs)

    if not use_deprecated:
        assert torch.equal(input, expected)
