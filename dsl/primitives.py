"""DSL primitive types and functions for `@infini_op` definitions.

These are used purely for type annotation and AST parsing — they have
no runtime behavior.  The function bodies serve as PyTorch-compatible
reference implementations for testing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    import torch

T = TypeVar("T")


# ---- Type annotations -----------------------------------------------------


class Tensor:
    """Annotates a tensor parameter with shape variables.

    Usage: ``input: Tensor["B", "H", "D"]``
    """

    def __class_getitem__(cls, item: Any) -> Any:
        return cls


class Scalar(Generic[T]):
    """Annotates a scalar parameter.

    Usage: ``eps: Scalar[float] = 1e-6``
    """

    pass


# ---- Elementwise functions -------------------------------------------------


def sqrt(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(x)


def rsqrt(x: torch.Tensor) -> torch.Tensor:
    return torch.rsqrt(x)


def exp(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x)


def log(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x)


def abs(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(x)


def neg(x: torch.Tensor) -> torch.Tensor:
    return -x


def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)


def gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x)


def silu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(x)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)


# ---- Reduction functions ---------------------------------------------------


def reduce_sum(
    x: torch.Tensor,
    dim: str | int = -1,
) -> torch.Tensor:
    return torch.sum(x, dim=-1, keepdim=True)


def reduce_mean(
    x: torch.Tensor,
    dim: str | int = -1,
) -> torch.Tensor:
    return torch.mean(x, dim=-1, keepdim=True)


def reduce_max(
    x: torch.Tensor,
    dim: str | int = -1,
) -> torch.Tensor:
    return torch.max(x, dim=-1, keepdim=True).values


def reduce_min(
    x: torch.Tensor,
    dim: str | int = -1,
) -> torch.Tensor:
    return torch.min(x, dim=-1, keepdim=True).values


# ---- Conditional -----------------------------------------------------------


def where(
    cond: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return torch.where(cond, a, b)


# ---- Type -------------------------------------------------------------------


def cast(x: torch.Tensor, dtype: Any) -> torch.Tensor:
    return x.to(dtype)


# ---- Clamp ------------------------------------------------------------------


def clamp(
    x: torch.Tensor,
    min: float | None = None,
    max: float | None = None,
) -> torch.Tensor:
    return torch.clamp(x, min=min, max=max)
