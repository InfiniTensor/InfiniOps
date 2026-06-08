#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import re
from functools import lru_cache


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_BASE_DIR = _REPO_ROOT / "src" / "base"
_TORCH_OPS_YAML = _REPO_ROOT / "scripts" / "torch_ops.yaml"

BASIC_MATH = "基础数学算子"
COMPARISON_LOGIC_BITWISE = "比较逻辑与位运算算子"
REDUCTION_STAT_SORT = "规约统计与排序算子"
LINALG = "矩阵计算与线性代数算子"
NEURAL_NETWORK = "神经网络常用算子"
TENSOR_ORGANIZATION = "索引聚合与张量组织算子"
FFT_SPECIAL = "FFT 与特殊函数算子"

CATEGORIES = (
    BASIC_MATH,
    COMPARISON_LOGIC_BITWISE,
    REDUCTION_STAT_SORT,
    LINALG,
    NEURAL_NETWORK,
    TENSOR_ORGANIZATION,
    FFT_SPECIAL,
)

CATEGORY_SLUGS = {
    BASIC_MATH: "basic-math",
    COMPARISON_LOGIC_BITWISE: "comparison-logic-bitwise",
    REDUCTION_STAT_SORT: "reduction-stat-sort",
    LINALG: "linalg",
    NEURAL_NETWORK: "nn",
    TENSOR_ORGANIZATION: "tensor-organization",
    FFT_SPECIAL: "fft-special",
}

_SPECIAL_OPS = frozenset(
    {
        "digamma",
        "erf",
        "erfc",
        "erfinv",
        "i0",
        "igamma",
        "igammac",
        "lgamma",
        "mvlgamma",
        "polygamma",
        "sinc",
        "xlogy",
        "xlogy_inplace",
    }
)

_COMPARISON_LOGIC_BITWISE_OPS = frozenset(
    {
        "eq",
        "eq_inplace",
        "fmax",
        "fmin",
        "ge",
        "ge_inplace",
        "greater",
        "greater_equal",
        "greater_equal_inplace",
        "greater_inplace",
        "gt",
        "gt_inplace",
        "isneginf",
        "isin",
        "isposinf",
        "le",
        "le_inplace",
        "less",
        "less_equal",
        "less_equal_inplace",
        "less_inplace",
        "lt",
        "lt_inplace",
        "maximum",
        "minimum",
        "ne",
        "ne_inplace",
        "not_equal",
        "not_equal_inplace",
        "signbit",
        "where",
    }
)

_REDUCTION_STAT_SORT_OPS = frozenset(
    {
        "all",
        "amax",
        "amin",
        "aminmax",
        "any",
        "argmax",
        "argmin",
        "argsort",
        "aten_logcumsumexp",
        "cummax",
        "cummin",
        "cumprod",
        "cumsum",
        "histc",
        "histogram",
        "kthvalue",
        "logcumsumexp",
        "logsumexp",
        "max",
        "mean",
        "median",
        "min",
        "mode",
        "msort",
        "nanmean",
        "nanmedian",
        "nanquantile",
        "nansum",
        "norm",
        "prod",
        "quantile",
        "renorm",
        "sort",
        "std",
        "sum",
        "topk",
        "var",
    }
)

_LINALG_OPS = frozenset(
    {
        "addbmm",
        "addmm",
        "addmv",
        "addr",
        "aten_addmm_activation",
        "aten_int_mm",
        "aten_scaled_mm",
        "baddbmm",
        "bmm",
        "cholesky",
        "cholesky_inverse",
        "cholesky_solve",
        "cross",
        "dot",
        "frobenius_norm",
        "gemm",
        "geqrf",
        "ger",
        "hspmm",
        "inner",
        "inverse",
        "kron",
        "linear",
        "lu_solve",
        "lu_unpack",
        "matmul",
        "matrix_power",
        "mm",
        "mv",
        "nuclear_norm",
        "orgqr",
        "ormqr",
        "outer",
        "qr",
        "slogdet",
        "sparse_sampled_addmm",
        "sspaddmm",
        "svd",
        "tensordot",
        "triangular_solve",
        "vdot",
    }
)

_NN_OPS = frozenset(
    {
        "add_rms_norm",
        "aten_add_relu",
        "aten_batch_norm_with_update",
        "aten_log_softmax",
        "aten_softmax",
        "batch_norm_elemt",
        "binary_cross_entropy",
        "binary_cross_entropy_backward",
        "causal_softmax",
        "elu",
        "elu_backward",
        "flash_attention",
        "gelu",
        "gelu_backward",
        "glu",
        "glu_backward",
        "hardshrink",
        "hardshrink_backward",
        "hardsigmoid",
        "hardsigmoid_backward",
        "hardswish",
        "hardtanh",
        "hardtanh_backward",
        "huber_loss",
        "huber_loss_backward",
        "leaky_relu",
        "leaky_relu_backward",
        "log_sigmoid",
        "log_sigmoid_backward",
        "log_sigmoid_forward",
        "log_softmax",
        "mish",
        "mse_loss",
        "mse_loss_backward",
        "multi_margin_loss",
        "multi_margin_loss_backward",
        "multilabel_margin_loss",
        "multilabel_margin_loss_backward",
        "multilabel_margin_loss_forward",
        "native_batch_norm",
        "nll_loss",
        "nll_loss2d",
        "nll_loss2d_backward",
        "nll_loss2d_forward",
        "nll_loss_backward",
        "nll_loss_forward",
        "reshape_and_cache",
        "rms_norm",
        "rotary_embedding",
        "rrelu_with_noise",
        "sigmoid",
        "sigmoid_backward",
        "silu",
        "silu_backward",
        "smooth_l1_loss",
        "smooth_l1_loss_backward",
        "soft_margin_loss",
        "soft_margin_loss_backward",
        "softmax",
        "softplus",
        "softplus_backward",
        "softshrink",
        "softshrink_backward",
        "swiglu",
        "tanh",
        "tanh_backward",
        "threshold",
        "threshold_backward",
    }
)

_TENSOR_ORGANIZATION_OPS = frozenset(
    {
        "arange",
        "bernoulli",
        "bernoulli_inplace",
        "bucketize",
        "cast",
        "cat",
        "col2im",
        "complex",
        "aten_convert_indices_from_coo_to_csr",
        "aten_convert_indices_from_csr_to_coo",
        "diag",
        "empty",
        "eye",
        "fill_inplace",
        "full",
        "gather",
        "im2col",
        "index",
        "index_add",
        "index_copy",
        "index_reduce",
        "index_select",
        "linspace",
        "logspace",
        "masked_fill_inplace",
        "masked_select",
        "multinomial",
        "narrow_copy",
        "nonzero",
        "nonzero_static",
        "normal",
        "ones",
        "rand",
        "randint",
        "randn",
        "randperm",
        "range",
        "scatter",
        "scatter_add",
        "scatter_reduce",
        "searchsorted",
        "set_inplace",
        "split_copy",
        "split_with_sizes_copy",
        "take",
        "take_along_dim",
        "tril",
        "triu",
        "unbind_copy",
        "zeros",
    }
)

_NN_PREFIXES = (
    "adaptive_avg_pool",
    "adaptive_max_pool",
    "aten_conv_",
    "aten_slow_conv",
    "aten_upsample_",
    "avg_pool",
    "cudnn_convolution",
    "fractional_max_pool",
    "max_pool",
    "max_unpool",
    "mkldnn_adaptive_avg_pool",
    "reflection_pad",
    "replication_pad",
    "slow_conv",
    "thnn_conv",
    "upsample_",
)

_CATEGORY_ALIASES = {
    "basic": BASIC_MATH,
    "basic_math": BASIC_MATH,
    "math": BASIC_MATH,
    "基础数学": BASIC_MATH,
    "基础数学算子": BASIC_MATH,
    "comparison": COMPARISON_LOGIC_BITWISE,
    "comparison_logic_bitwise": COMPARISON_LOGIC_BITWISE,
    "compare": COMPARISON_LOGIC_BITWISE,
    "logic": COMPARISON_LOGIC_BITWISE,
    "bitwise": COMPARISON_LOGIC_BITWISE,
    "比较逻辑": COMPARISON_LOGIC_BITWISE,
    "位运算": COMPARISON_LOGIC_BITWISE,
    "比较逻辑与位运算算子": COMPARISON_LOGIC_BITWISE,
    "reduction": REDUCTION_STAT_SORT,
    "reduce": REDUCTION_STAT_SORT,
    "statistics": REDUCTION_STAT_SORT,
    "stat": REDUCTION_STAT_SORT,
    "sort": REDUCTION_STAT_SORT,
    "reduction_stat_sort": REDUCTION_STAT_SORT,
    "规约统计": REDUCTION_STAT_SORT,
    "排序": REDUCTION_STAT_SORT,
    "规约统计与排序算子": REDUCTION_STAT_SORT,
    "linalg": LINALG,
    "linear_algebra": LINALG,
    "matrix": LINALG,
    "matrix_linalg": LINALG,
    "矩阵计算": LINALG,
    "线性代数": LINALG,
    "矩阵计算与线性代数算子": LINALG,
    "nn": NEURAL_NETWORK,
    "neural": NEURAL_NETWORK,
    "neural_network": NEURAL_NETWORK,
    "network": NEURAL_NETWORK,
    "神经网络": NEURAL_NETWORK,
    "神经网络常用算子": NEURAL_NETWORK,
    "tensor": TENSOR_ORGANIZATION,
    "tensor_organization": TENSOR_ORGANIZATION,
    "index": TENSOR_ORGANIZATION,
    "indexing": TENSOR_ORGANIZATION,
    "organization": TENSOR_ORGANIZATION,
    "索引聚合": TENSOR_ORGANIZATION,
    "张量组织": TENSOR_ORGANIZATION,
    "索引聚合与张量组织算子": TENSOR_ORGANIZATION,
    "fft": FFT_SPECIAL,
    "special": FFT_SPECIAL,
    "fft_special": FFT_SPECIAL,
    "特殊函数": FFT_SPECIAL,
    "fft与特殊函数算子": FFT_SPECIAL,
    "fft_与特殊函数算子": FFT_SPECIAL,
    "fft 与特殊函数算子": FFT_SPECIAL,
}

_ALL_CATEGORY_SELECTORS = frozenset({"*", "all", "全部", "全部算子"})


def public_op_name(aten_name: str) -> str:
    public_name = aten_name

    if public_name.startswith("_"):
        public_name = f"aten{public_name}"

    if public_name.endswith("_") and not public_name.endswith("__"):
        public_name = public_name[:-1] + "_inplace"

    return public_name


def category_for_operator(operator: str | None) -> str | None:
    if not operator:
        return None

    operator = public_op_name(operator)

    if (
        operator.startswith("fft_")
        or operator.startswith("aten_fft_")
        or operator.startswith("special_")
        or operator in _SPECIAL_OPS
    ):
        return FFT_SPECIAL

    if operator in _NN_OPS or operator.startswith(_NN_PREFIXES):
        return NEURAL_NETWORK

    if operator.startswith("linalg_") or operator.startswith("aten_linalg_"):
        return LINALG

    if operator in _LINALG_OPS:
        return LINALG

    if operator.startswith("bitwise_") or operator.startswith("logical_"):
        return COMPARISON_LOGIC_BITWISE

    if operator in _COMPARISON_LOGIC_BITWISE_OPS:
        return COMPARISON_LOGIC_BITWISE

    if operator in _REDUCTION_STAT_SORT_OPS:
        return REDUCTION_STAT_SORT

    if operator in _TENSOR_ORGANIZATION_OPS:
        return TENSOR_ORGANIZATION

    return BASIC_MATH


def inventory_operator_name(
    operator: str | None, aten_name: str | None = None
) -> str | None:
    semantic_name = aten_name or operator

    if semantic_name is None:
        return None

    return public_op_name(semantic_name)


def load_operator_inventory(
    base_dir: pathlib.Path | None = None,
    torch_ops_yaml: pathlib.Path | None = None,
) -> list[dict[str, str]]:
    base_dir = base_dir or _BASE_DIR
    torch_ops_yaml = torch_ops_yaml or _TORCH_OPS_YAML
    inventory = {}

    for path in sorted(base_dir.glob("*.h")):
        inventory[path.stem] = {
            "operator": path.stem,
            "category": category_for_operator(path.stem),
            "source": "native",
        }

    for line in torch_ops_yaml.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue

        operator = public_op_name(stripped[2:])
        inventory.setdefault(
            operator,
            {
                "operator": operator,
                "category": category_for_operator(operator),
                "source": "torch-generated",
            },
        )

    return [inventory[name] for name in sorted(inventory)]


@lru_cache
def native_operator_names() -> frozenset[str]:
    return frozenset(path.stem for path in _BASE_DIR.glob("*.h"))


def is_native_operator(operator: str | None) -> bool:
    return operator in native_operator_names()


def normalize_category_selector(selector: str) -> str | None:
    key = _normalize_selector(selector)

    if key in _ALL_CATEGORY_SELECTORS:
        return None

    category = _CATEGORY_ALIASES.get(key)

    if category is None:
        valid = ", ".join(CATEGORY_SLUGS.values())
        raise ValueError(
            f"unknown operator category {selector!r}; valid slugs: {valid}"
        )

    return category


def normalize_category_selectors(
    selectors: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    selected = []
    seen = set()

    for selector in selectors:
        category = normalize_category_selector(selector)

        if category is None:
            return ()

        if category not in seen:
            selected.append(category)
            seen.add(category)

    return tuple(selected)


def category_slug(category: str) -> str:
    return CATEGORY_SLUGS[category]


def selector_slug(selector: str) -> str:
    category = normalize_category_selector(selector)

    if category is None:
        return "all"

    return category_slug(category)


def selectors_slug(selectors: list[str] | tuple[str, ...]) -> str:
    categories = normalize_category_selectors(selectors)

    if not categories:
        return "all"

    return "-".join(category_slug(category) for category in categories)


def category_help() -> str:
    return ", ".join(f"{slug}={category}" for category, slug in CATEGORY_SLUGS.items())


def _normalize_selector(selector: str) -> str:
    normalized = selector.strip().lower()
    normalized = normalized.replace("-", "_")
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve InfiniOps operator category names and aliases."
    )
    parser.add_argument(
        "--slug",
        nargs="+",
        metavar="CATEGORY",
        help="Print a filesystem-friendly slug for one or more selectors.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print accepted category slugs and display names.",
    )
    args = parser.parse_args()

    if args.list:
        for category in CATEGORIES:
            print(f"{CATEGORY_SLUGS[category]}\t{category}")

    if args.slug:
        try:
            print(selectors_slug(tuple(args.slug)))
        except ValueError as exc:
            parser.exit(status=2, message=f"{parser.prog}: error: {exc}\n")


if __name__ == "__main__":
    main()
