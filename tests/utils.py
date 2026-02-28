import contextlib
import dataclasses
from collections.abc import Callable

import torch


@dataclasses.dataclass
class Payload:
    func: Callable

    ref: Callable

    args: tuple

    kwargs: dict

    rtol: float = 1e-5

    atol: float = 1e-8


def get_available_devices():
    devices = []

    if torch.cuda.is_available():
        devices.append("cuda")

    if hasattr(torch, "mlu") and torch.mlu.is_available():
        devices.append("mlu")

    return tuple(devices)


with contextlib.suppress(ImportError, ModuleNotFoundError):
    import torch_mlu  # noqa: F401


def empty_strided(shape, strides, *, dtype=None, device=None):
    if strides is None:
        return torch.empty(shape, dtype=dtype, device=device)

    return torch.empty_strided(shape, strides, dtype=dtype, device=device)


def randn_strided(shape, strides, *, dtype=None, device=None):
    output = empty_strided(shape, strides, dtype=dtype, device=device)

    output.as_strided(
        (output.untyped_storage().size() // output.element_size(),), (1,)
    ).normal_()

    return output
