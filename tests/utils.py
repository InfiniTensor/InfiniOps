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
    devices = ["cpu"]

    if torch.cuda.is_available():
        devices.append("cuda")

    if hasattr(torch, "mlu") and torch.mlu.is_available():
        devices.append("mlu")

    if hasattr(torch, "musa") and torch.musa.is_available():
        devices.append("musa")

    if hasattr(torch, "npu") and torch.npu.is_available():
        devices.append("npu")

    return tuple(devices)


with contextlib.suppress(ImportError, ModuleNotFoundError):
    import torch_mlu  # noqa: F401

with contextlib.suppress(ImportError, ModuleNotFoundError):
    import torch_npu  # noqa: F401


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


def rand_strided(shape, strides, *, dtype=None, device=None):
    output = empty_strided(shape, strides, dtype=dtype, device=device)

    output.as_strided(
        (output.untyped_storage().size() // output.element_size(),), (1,)
    ).uniform_(0, 1)

    return output


def randint_strided(low, high, shape, strides, *, dtype=None, device=None):
    output = empty_strided(shape, strides, dtype=dtype, device=device)

    output.as_strided(
        (output.untyped_storage().size() // output.element_size(),), (1,)
    ).random_(low, high)

    return output


_STREAM_ACCESSORS = {
    "npu": ("npu", "npu_stream"),
    "cuda": ("cuda", "cuda_stream"),
    "mlu": ("mlu", "mlu_stream"),
    "musa": ("musa", "musa_stream"),
}


def get_stream(device):
    """Return the raw stream handle for `device`, or 0 for CPU.

    Prefers the device-specific API (e.g. `torch.npu.current_stream`) over
    `torch.accelerator.current_stream`: on torch 2.9 + `torch_npu`, the two
    return different stream handles, and kernels submitted on one race
    against PyTorch work on the other when an operator caches its executor
    (observed on `Gemm`/`Linear`).
    """
    if isinstance(device, torch.device):
        device = device.type

    if isinstance(device, str) and ":" in device:
        device = device.split(":")[0]

    if device == "cpu":
        return 0

    if device in _STREAM_ACCESSORS:
        mod_name, attr = _STREAM_ACCESSORS[device]
        mod = getattr(torch, mod_name, None)

        if mod is not None and hasattr(mod, "current_stream"):
            return getattr(mod.current_stream(), attr)

    if hasattr(torch, "accelerator") and hasattr(torch.accelerator, "current_stream"):
        stream = torch.accelerator.current_stream()

        for attr in ("npu_stream", "cuda_stream", "mlu_stream", "musa_stream"):
            if hasattr(stream, attr):
                return getattr(stream, attr)

    return 0


def clone_strided(input):
    output = empty_strided(
        input.size(), input.stride(), dtype=input.dtype, device=input.device
    )

    as_strided_args = (output.untyped_storage().size() // output.element_size(),), (1,)

    output.as_strided(*as_strided_args).copy_(input.as_strided(*as_strided_args))

    return output
