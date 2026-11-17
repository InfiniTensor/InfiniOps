from __future__ import annotations

import functools
from contextlib import AbstractContextManager

import infini.ops
import torch

try:
    from torch.utils._python_dispatch import TorchDispatchMode
except ImportError:
    TorchDispatchMode = None


__all__ = ("patch",)

_PATCHED_OPS = ("rms_norm",)
_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def patch():
    return _TorchPatch()


class _TorchPatch(AbstractContextManager):
    def __enter__(self):
        _check_runtime()
        self._mode = _InfiniTorchDispatchMode()
        self._mode.__enter__()
        self._original_rms_norm = torch.nn.functional.rms_norm
        torch.nn.functional.rms_norm = _wrap_functional_rms_norm(
            self._original_rms_norm
        )

        return None

    def __exit__(self, exc_type, exc_value, traceback):
        torch.nn.functional.rms_norm = self._original_rms_norm

        return self._mode.__exit__(exc_type, exc_value, traceback)


class _InfiniTorchDispatchMode(TorchDispatchMode if TorchDispatchMode else object):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        if _is_aten_rms_norm(func):
            return _rms_norm(func, args, kwargs, fused=False)

        if _is_aten_fused_rms_norm(func):
            return _rms_norm(func, args, kwargs, fused=True)

        return func(*args, **kwargs)


def _wrap_functional_rms_norm(original):
    @functools.wraps(original)
    def wrapper(input, normalized_shape, weight=None, eps=None):
        if _can_use_infini_rms_norm(input, normalized_shape, weight):
            return _call_infini_rms_norm(input, normalized_shape, weight, eps)

        return original(input, normalized_shape, weight=weight, eps=eps)

    return wrapper


def _check_runtime():
    if TorchDispatchMode is None:
        raise RuntimeError(
            "`TorchDispatchMode` is not available in this PyTorch build."
        )

    if not hasattr(torch.nn.functional, "rms_norm"):
        raise RuntimeError(
            "`torch.nn.functional.rms_norm` is not available in this PyTorch build."
        )

    if not (_has_aten_op("rms_norm") or _has_aten_op("_fused_rms_norm")):
        raise RuntimeError(
            "`aten::rms_norm` is not available in this PyTorch build."
        )


def _has_aten_op(name):
    return hasattr(torch.ops.aten, name)


def _is_aten_rms_norm(func):
    return _has_aten_op("rms_norm") and func is torch.ops.aten.rms_norm.default


def _is_aten_fused_rms_norm(func):
    return (
        _has_aten_op("_fused_rms_norm")
        and func is torch.ops.aten._fused_rms_norm.default
    )


def _rms_norm(func, args, kwargs, *, fused):
    input, normalized_shape, weight, eps = _parse_rms_norm_args(args, kwargs)

    if not _can_use_infini_rms_norm(input, normalized_shape, weight):
        return func(*args, **kwargs)

    out = _call_infini_rms_norm(input, normalized_shape, weight, eps)

    if fused:
        return out, _rms_norm_rstd(input, normalized_shape, eps)

    return out


def _call_infini_rms_norm(input, normalized_shape, weight, eps):
    eps = 1e-6 if eps is None else eps
    out = torch.empty_like(input)
    infini.ops.rms_norm(
        input,
        weight,
        eps,
        out,
        implementation_index=0,
        stream=torch.cuda.current_stream(input.device).cuda_stream,
    )

    return out


def _parse_rms_norm_args(args, kwargs):
    input = _arg(args, kwargs, 0, "input")
    normalized_shape = _arg(args, kwargs, 1, "normalized_shape")
    weight = _arg(args, kwargs, 2, "weight", None)
    eps = _arg(args, kwargs, 3, "eps", None)

    return input, normalized_shape, weight, eps


def _arg(args, kwargs, index, name, default=None):
    if len(args) > index:
        return args[index]

    return kwargs.get(name, default)


def _can_use_infini_rms_norm(input, normalized_shape, weight):
    if not isinstance(input, torch.Tensor) or not isinstance(weight, torch.Tensor):
        return False

    if input.device.type != "cuda" or weight.device != input.device:
        return False

    if input.layout != torch.strided or weight.layout != torch.strided:
        return False

    if input.dtype not in _SUPPORTED_DTYPES or weight.dtype != input.dtype:
        return False

    if _is_metax_torch() and input.dtype != torch.float32:
        return False

    shape = _normalized_shape_tuple(normalized_shape)

    if shape != tuple(input.shape[-1:]):
        return False

    return weight.ndim == 1 and tuple(weight.shape) == shape


def _is_metax_torch():
    return "metax" in torch.__version__.lower()


def _normalized_shape_tuple(normalized_shape):
    if isinstance(normalized_shape, int):
        return (normalized_shape,)

    return tuple(int(dim) for dim in normalized_shape)


def _rms_norm_rstd(input, normalized_shape, eps):
    eps = 1e-6 if eps is None else eps
    normalized_shape = _normalized_shape_tuple(normalized_shape)
    dim = tuple(range(input.ndim - len(normalized_shape), input.ndim))
    square_mean = torch.mean(input.float() * input.float(), dim=dim, keepdim=True)

    return torch.rsqrt(square_mean + eps)
