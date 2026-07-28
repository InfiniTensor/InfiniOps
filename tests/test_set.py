import warnings

import infini.ops
import pytest
import torch

from tests.utils import get_stream


@pytest.mark.parametrize("storage_kind", ("typed", "untyped"))
@pytest.mark.parametrize("with_metadata", (False, True))
def test_set_storage(storage_kind, with_metadata, device, implementation_index):
    source = torch.arange(8, dtype=torch.float32, device=device)
    input = torch.empty(0, dtype=source.dtype, device=device)

    if storage_kind == "typed":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            storage = source.storage()
    else:
        storage = source.untyped_storage()

    args = (input, storage, 1, (2, 2), (2, 1)) if with_metadata else (input, storage)
    result = infini.ops.set(
        *args,
        stream=get_stream(device),
        implementation_index=implementation_index,
    )

    assert result is None
