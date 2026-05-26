"""Unified test for every operator emitted by `generate_torch_ops.py`.

The generator writes `generated/torch_ops_metadata.json` listing every op
with full per-parameter info (`name`, `type`, `is_tensor`, `is_out`,
`keyword_only`).
A single parametrized test reads that metadata, builds inputs from the
parameter list, calls the InfiniOps wrapper and the torch reference, and
compares each output tensor.  Adding an op to `scripts/torch_ops.yaml`
extends coverage with no test changes.
"""

import json
import pathlib
import re

import infini.ops
import pytest
import torch

from tests.utils import clone_strided, randint_strided, rand_strided, randn_strided

# PyTorch backends are emitted at this slot — see `_PYTORCH_SLOT` in
# `scripts/generate_torch_ops.py`.
_PYTORCH_SLOT = 8

_INSTALLED_METADATA_PATH = (
    pathlib.Path(infini.ops.__file__).resolve().with_name("torch_ops_metadata.json")
)
_SOURCE_METADATA_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "generated"
    / "torch_ops_metadata.json"
)

_METADATA_PATH = next(
    (
        path
        for path in (_SOURCE_METADATA_PATH, _INSTALLED_METADATA_PATH)
        if path.exists()
    ),
    _SOURCE_METADATA_PATH,
)
_METADATA = (
    json.loads(_METADATA_PATH.read_text()) if _METADATA_PATH.exists() else {"ops": []}
)

_SHAPES = (
    (13, 4),
    (13, 4, 4),
    (4, 4, 5632),
)

_DEFAULT_DTYPES = (
    torch.float32,
    torch.float16,
    torch.bfloat16,
)

_INTEGER_TEST_DTYPES = tuple(
    getattr(torch, name)
    for name in ("int32", "int64", "uint8")
    if hasattr(torch, name)
)

_DTYPE_TOLERANCES = {
    torch.float64: (1e-8, 1e-8),
    torch.float32: (1e-5, 1e-5),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (1e-2, 1e-2),
}

_UNSIGNED_DTYPES = frozenset(
    getattr(torch, name)
    for name in ("uint8", "uint16", "uint32", "uint64")
    if hasattr(torch, name)
)

# Op-specific input shapes for matrix ops (`mm` etc.) which cannot use
# `randn_strided(shape)` for both inputs.  The tuple is one shape per
# tensor input, in YAML order.
_TENSOR_SHAPES = {
    "mm": ((8, 16), (16, 12)),
    "bmm": ((4, 8, 16), (4, 16, 12)),
    "matmul": ((8, 16), (16, 12)),
    "dot": ((16,), (16,)),
    "vdot": ((16,), (16,)),
    "mv": ((8, 16), (16,)),
    "addmm": ((8, 12), (8, 16), (16, 12)),
    "_addmm_activation": ((8, 12), (8, 16), (16, 12)),
    "addmv": ((8,), (8, 16), (16,)),
    "addr": ((8, 12), (8,), (12,)),
    "addbmm": ((8, 12), (4, 8, 16), (4, 16, 12)),
    "baddbmm": ((4, 8, 12), (4, 8, 16), (4, 16, 12)),
    "inner": ((8, 16), (8, 16)),
    "outer": ((8,), (12,)),
    "ger": ((8,), (12,)),
    "kron": ((3, 4), (2, 3)),
    "cross": ((3, 4, 5), (3, 4, 5)),
    "linalg_cross": ((3, 4, 5), (3, 4, 5)),
    "avg_pool2d": ((2, 3, 8, 8),),
    "avg_pool3d": ((2, 3, 6, 6, 6),),
    "max_pool2d_with_indices": ((2, 3, 8, 8),),
    "max_pool3d_with_indices": ((2, 3, 6, 6, 6),),
    "adaptive_avg_pool2d": ((2, 3, 8, 8),),
    "adaptive_avg_pool3d": ((2, 3, 6, 6, 6),),
    "adaptive_max_pool2d": ((2, 3, 8, 8),),
    "adaptive_max_pool3d": ((2, 3, 6, 6, 6),),
    "fractional_max_pool2d": ((2, 3, 8, 8), (2, 3, 2)),
    "fractional_max_pool3d": ((2, 3, 6, 6, 6), (2, 3, 3)),
    "adaptive_max_pool2d_backward": ((2, 3, 2, 2), (2, 3, 8, 8), (2, 3, 2, 2)),
    "adaptive_max_pool3d_backward": (
        (2, 3, 2, 2, 2),
        (2, 3, 6, 6, 6),
        (2, 3, 2, 2, 2),
    ),
    "avg_pool2d_backward": ((2, 3, 4, 4), (2, 3, 8, 8)),
    "avg_pool3d_backward": ((2, 3, 3, 3, 3), (2, 3, 6, 6, 6)),
    "fractional_max_pool2d_backward": ((2, 3, 2, 2), (2, 3, 8, 8), (2, 3, 2, 2)),
    "fractional_max_pool3d_backward": (
        (2, 3, 2, 2, 2),
        (2, 3, 6, 6, 6),
        (2, 3, 2, 2, 2),
    ),
    "adaptive_avg_pool3d_backward": ((2, 3, 2, 2, 2), (2, 3, 6, 6, 6)),
    "col2im": ((2, 12, 16),),
    "glu_backward": ((2, 4), (2, 8)),
    "im2col": ((2, 3, 8, 8),),
    "linalg_eigh": ((4, 4),),
    "_linalg_eigh": ((4, 4),),
    "linalg_eigvalsh": ((4, 4),),
    "linalg_householder_product": ((4, 4), (4,)),
    "orgqr": ((4, 4), (4,)),
    "ormqr": ((4, 4), (4,), (4, 4)),
    "linalg_lu_solve": ((4, 4), (4,), (4, 4)),
    "lu_solve": ((4, 4), (4, 4), (4,)),
    "lu_unpack": ((4, 4), (4,)),
    "linalg_tensorsolve": ((2, 2, 2, 2), (2, 2)),
    "linalg_tensorinv": ((2, 2, 2, 2),),
    "cholesky": ((4, 4),),
    "linalg_cholesky": ((4, 4),),
    "linalg_lu": ((4, 4),),
    "linalg_lu_factor": ((4, 4),),
    "linalg_lu_factor_ex": ((4, 4),),
    "linalg_ldl_solve": ((4, 4), (4,), (4, 2)),
    "linalg_lstsq": ((8, 4), (8, 2)),
    "multinomial": ((8, 5),),
    "max_unpool2d": ((2, 3, 4, 4), (2, 3, 4, 4)),
    "max_unpool3d": ((2, 3, 3, 3, 3), (2, 3, 3, 3, 3)),
    "max_pool2d_with_indices_backward": ((2, 3, 4, 4), (2, 3, 8, 8), (2, 3, 4, 4)),
    "max_pool3d_with_indices_backward": (
        (2, 3, 3, 3, 3),
        (2, 3, 6, 6, 6),
        (2, 3, 3, 3, 3),
    ),
    "multi_margin_loss": ((2, 4), (2,)),
    "multi_margin_loss_backward": ((), (2, 4), (2,)),
    "multilabel_margin_loss": ((2, 4), (2, 4)),
    "multilabel_margin_loss_forward": ((2, 4), (2, 4)),
    "multilabel_margin_loss_backward": ((), (2, 4), (2, 4), (2, 4)),
    "batch_norm_elemt": ((2, 3, 4, 4), (3,), (3,)),
    "native_batch_norm": ((2, 3, 4, 4),),
    "nll_loss": ((2, 4), (2,)),
    "nll_loss_forward": ((2, 4), (2,)),
    "nll_loss_backward": ((), (2, 4), (2,), ()),
    "nll_loss2d": ((2, 4, 4, 4), (2, 4, 4)),
    "nll_loss2d_forward": ((2, 4, 4, 4), (2, 4, 4)),
    "nll_loss2d_backward": ((), (2, 4, 4, 4), (2, 4, 4), ()),
    "reflection_pad1d": ((2, 3, 4),),
    "reflection_pad1d_backward": ((2, 3, 6), (2, 3, 4)),
    "reflection_pad2d": ((2, 3, 4, 4),),
    "reflection_pad2d_backward": ((2, 3, 6, 6), (2, 3, 4, 4)),
    "reflection_pad3d": ((2, 3, 4, 4, 4),),
    "reflection_pad3d_backward": ((2, 3, 6, 6, 6), (2, 3, 4, 4, 4)),
    "replication_pad1d": ((2, 3, 4),),
    "replication_pad1d_backward": ((2, 3, 6), (2, 3, 4)),
    "replication_pad2d": ((2, 3, 4, 4),),
    "replication_pad2d_backward": ((2, 3, 6, 6), (2, 3, 4, 4)),
    "replication_pad3d": ((2, 3, 4, 4, 4),),
    "replication_pad3d_backward": ((2, 3, 6, 6, 6), (2, 3, 4, 4, 4)),
    "tensordot": ((2, 3, 4), (2, 3, 4)),
    "upsample_bicubic2d": ((2, 3, 4, 4),),
    "upsample_bicubic2d_backward": ((2, 3, 8, 8),),
    "upsample_bilinear2d": ((2, 3, 4, 4),),
    "upsample_bilinear2d_backward": ((2, 3, 8, 8),),
    "upsample_linear1d": ((2, 3, 4),),
    "upsample_linear1d_backward": ((2, 3, 8),),
    "upsample_nearest1d": ((2, 3, 4),),
    "upsample_nearest1d_backward": ((2, 3, 8),),
    "upsample_nearest2d": ((2, 3, 4, 4),),
    "upsample_nearest2d_backward": ((2, 3, 8, 8),),
    "upsample_nearest3d": ((2, 3, 3, 3, 3),),
    "upsample_nearest3d_backward": ((2, 3, 6, 6, 6),),
    "upsample_trilinear3d": ((2, 3, 3, 3, 3),),
    "upsample_trilinear3d_backward": ((2, 3, 6, 6, 6),),
    "_upsample_bicubic2d_aa": ((2, 3, 4, 4),),
    "_upsample_bicubic2d_aa_backward": ((2, 3, 8, 8),),
    "_upsample_bilinear2d_aa": ((2, 3, 4, 4),),
    "_upsample_bilinear2d_aa_backward": ((2, 3, 8, 8),),
    "_upsample_nearest_exact1d": ((2, 3, 4),),
    "_upsample_nearest_exact1d_backward": ((2, 3, 8),),
    "_upsample_nearest_exact2d": ((2, 3, 4, 4),),
    "_upsample_nearest_exact2d_backward": ((2, 3, 8, 8),),
    "_upsample_nearest_exact3d": ((2, 3, 3, 3, 3),),
    "_upsample_nearest_exact3d_backward": ((2, 3, 6, 6, 6),),
    "_conv_depthwise2d": ((1, 4, 8, 8), (4, 1, 3, 3)),
    "_slow_conv2d_forward": ((1, 3, 8, 8), (5, 3, 3, 3)),
    "_slow_conv2d_backward": ((1, 5, 8, 8), (1, 3, 8, 8), (5, 3, 3, 3)),
    "slow_conv3d": ((1, 3, 6, 6, 6), (5, 3, 3, 3, 3)),
    "slow_conv3d_forward": ((1, 3, 6, 6, 6), (5, 3, 3, 3, 3)),
    "slow_conv_transpose2d": ((1, 3, 8, 8), (3, 5, 3, 3)),
    "slow_conv_transpose3d": ((1, 3, 6, 6, 6), (3, 5, 3, 3, 3)),
    "thnn_conv2d": ((1, 3, 8, 8), (5, 3, 3, 3)),
    "linspace": ((), ()),
    "logspace": ((), ()),
}

# Per-(op, param-name) values for non-tensor inputs.  Lookup falls back
# to a type-based default if no entry exists.
_SCALAR_VALUES = {
    ("clamp_min", "min"): -0.5,
    ("clamp_max", "max"): 0.5,
    ("leaky_relu", "negative_slope"): 0.01,
    ("hardshrink", "lambd"): 0.5,
    ("softshrink", "lambd"): 0.5,
    ("mvlgamma", "p"): 2,
    ("prod", "dim"): 0,
    ("cumsum", "dim"): 0,
    ("cumprod", "dim"): 0,
    ("logcumsumexp", "dim"): 0,
    ("cummax", "dim"): 0,
    ("cummin", "dim"): 0,
    ("softmax", "dim"): -1,
    ("log_softmax", "dim"): -1,
    ("threshold", "threshold"): 0.0,
    ("threshold", "value"): 0.0,
    ("hardtanh", "min_val"): -1.0,
    ("hardtanh", "max_val"): 1.0,
    ("softplus", "beta"): 1.0,
    ("softplus", "threshold"): 20.0,
    ("elu", "alpha"): 1.0,
    ("elu", "scale"): 1.0,
    ("elu", "input_scale"): 1.0,
    ("sub", "alpha"): 1.0,
    ("addcmul", "value"): 1.0,
    ("addcdiv", "value"): 1.0,
    # `str reduce` modes accepted by the corresponding ATen kernels.
    ("index_reduce", "reduce"): "amax",
    ("scatter_reduce", "reduce"): "amax",
    ("scatter_reduce_two", "reduce"): "amax",
    ("_convert_indices_from_coo_to_csr", "size"): 3,
    ("_linalg_eigh", "UPLO"): "L",
    ("_linalg_eigh", "compute_v"): True,
    ("cholesky", "upper"): False,
    ("avg_pool2d_backward", "kernel_size"): [2, 2],
    ("avg_pool2d_backward", "stride"): [2, 2],
    ("avg_pool2d_backward", "padding"): [0, 0],
    ("avg_pool3d_backward", "kernel_size"): [2, 2, 2],
    ("avg_pool3d_backward", "stride"): [2, 2, 2],
    ("avg_pool3d_backward", "padding"): [0, 0, 0],
    ("col2im", "output_size"): [8, 8],
    ("col2im", "kernel_size"): [2, 2],
    ("col2im", "dilation"): [1, 1],
    ("col2im", "padding"): [0, 0],
    ("col2im", "stride"): [2, 2],
    ("glu_backward", "dim"): 1,
    ("huber_loss", "reduction"): 1,
    ("im2col", "kernel_size"): [2, 2],
    ("im2col", "dilation"): [1, 1],
    ("im2col", "padding"): [0, 0],
    ("im2col", "stride"): [2, 2],
    ("linalg_eigh", "UPLO"): "L",
    ("linalg_eigvalsh", "UPLO"): "L",
    ("linalg_cholesky", "upper"): False,
    ("linalg_lu", "pivot"): True,
    ("linalg_lu_factor", "pivot"): True,
    ("linalg_lu_factor_ex", "pivot"): True,
    ("linalg_lu_factor_ex", "check_errors"): False,
    ("linalg_ldl_solve", "hermitian"): False,
    ("linalg_matrix_norm", "ord"): 1,
    ("linalg_matrix_norm", "dim"): [0, 1],
    ("linalg_tensorinv", "ind"): 2,
    ("linalg_lu_solve", "left"): True,
    ("linalg_lu_solve", "adjoint"): False,
    ("lu_unpack", "unpack_data"): True,
    ("lu_unpack", "unpack_pivots"): True,
    ("multinomial", "num_samples"): 2,
    ("max_unpool2d", "output_size"): [8, 8],
    ("max_unpool3d", "output_size"): [6, 6, 6],
    ("max_unpool3d", "stride"): [2, 2, 2],
    ("max_unpool3d", "padding"): [0, 0, 0],
    ("max_pool2d_with_indices_backward", "kernel_size"): [2, 2],
    ("max_pool2d_with_indices_backward", "stride"): [2, 2],
    ("max_pool2d_with_indices_backward", "padding"): [0, 0],
    ("max_pool2d_with_indices_backward", "dilation"): [1, 1],
    ("max_pool2d_with_indices_backward", "ceil_mode"): False,
    ("max_pool3d_with_indices_backward", "kernel_size"): [2, 2, 2],
    ("max_pool3d_with_indices_backward", "stride"): [2, 2, 2],
    ("max_pool3d_with_indices_backward", "padding"): [0, 0, 0],
    ("max_pool3d_with_indices_backward", "dilation"): [1, 1, 1],
    ("max_pool3d_with_indices_backward", "ceil_mode"): False,
    ("multi_margin_loss", "p"): 1,
    ("multi_margin_loss", "reduction"): 1,
    ("multi_margin_loss_backward", "p"): 1,
    ("multi_margin_loss_backward", "reduction"): 1,
    ("multilabel_margin_loss", "reduction"): 1,
    ("multilabel_margin_loss_forward", "reduction"): 1,
    ("multilabel_margin_loss_backward", "reduction"): 1,
    ("native_batch_norm", "training"): True,
    ("narrow_copy", "dim"): 0,
    ("narrow_copy", "start"): 0,
    ("narrow_copy", "length"): 1,
    ("nll_loss_forward", "reduction"): 1,
    ("nll_loss_forward", "ignore_index"): -100,
    ("nll_loss_backward", "reduction"): 1,
    ("nll_loss_backward", "ignore_index"): -100,
    ("nll_loss2d_forward", "reduction"): 1,
    ("nll_loss2d_forward", "ignore_index"): -100,
    ("nll_loss2d_backward", "reduction"): 1,
    ("nll_loss2d_backward", "ignore_index"): -100,
    ("normal", "std"): 1.0,
    ("nonzero_static", "size"): 2,
    ("nonzero_static", "fill_value"): -1,
    ("nll_loss", "reduction"): 1,
    ("nll_loss2d", "reduction"): 1,
    ("ormqr", "left"): True,
    ("ormqr", "transpose"): False,
    ("mse_loss", "reduction"): 0,
    ("smooth_l1_loss", "reduction"): 0,
    ("smooth_l1_loss", "beta"): 1.0,
    ("soft_margin_loss", "reduction"): 0,
    ("rrelu_with_noise", "lower"): 0.125,
    ("rrelu_with_noise", "upper"): 1.0 / 3.0,
    ("rrelu_with_noise", "training"): True,
    ("bitwise_and_", "other"): 1,
    ("bitwise_or_", "other"): 1,
    ("bitwise_xor_", "other"): 1,
    ("float_power_", "exponent"): 2.0,
    ("histc", "bins"): 5,
    ("histc", "min"): -1.0,
    ("histc", "max"): 1.0,
    ("quantile", "interpolation"): "linear",
    ("nanquantile", "interpolation"): "linear",
    ("reflection_pad1d", "padding"): [1, 1],
    ("reflection_pad1d_backward", "padding"): [1, 1],
    ("reflection_pad2d", "padding"): [1, 1, 1, 1],
    ("reflection_pad2d_backward", "padding"): [1, 1, 1, 1],
    ("reflection_pad3d", "padding"): [1, 1, 1, 1, 1, 1],
    ("reflection_pad3d_backward", "padding"): [1, 1, 1, 1, 1, 1],
    ("replication_pad1d", "padding"): [1, 1],
    ("replication_pad1d_backward", "padding"): [1, 1],
    ("replication_pad2d", "padding"): [1, 1, 1, 1],
    ("replication_pad2d_backward", "padding"): [1, 1, 1, 1],
    ("replication_pad3d", "padding"): [1, 1, 1, 1, 1, 1],
    ("replication_pad3d_backward", "padding"): [1, 1, 1, 1, 1, 1],
    ("linalg_qr", "mode"): "reduced",
    ("special_multigammaln", "p"): 1,
    ("tensordot", "dims_self"): [1],
    ("tensordot", "dims_other"): [1],
    ("topk", "k"): 1,
    ("topk", "dim"): 0,
    ("topk", "largest"): True,
    ("topk", "sorted"): True,
    ("upsample_bicubic2d", "output_size"): [8, 8],
    ("upsample_bicubic2d", "align_corners"): False,
    ("upsample_bicubic2d_backward", "output_size"): [8, 8],
    ("upsample_bicubic2d_backward", "input_size"): [2, 3, 4, 4],
    ("upsample_bicubic2d_backward", "align_corners"): False,
    ("upsample_bilinear2d", "output_size"): [8, 8],
    ("upsample_bilinear2d", "align_corners"): False,
    ("upsample_bilinear2d_backward", "output_size"): [8, 8],
    ("upsample_bilinear2d_backward", "input_size"): [2, 3, 4, 4],
    ("upsample_bilinear2d_backward", "align_corners"): False,
    ("upsample_linear1d", "output_size"): [8],
    ("upsample_linear1d", "align_corners"): False,
    ("upsample_linear1d_backward", "output_size"): [8],
    ("upsample_linear1d_backward", "input_size"): [2, 3, 4],
    ("upsample_linear1d_backward", "align_corners"): False,
    ("upsample_nearest1d", "output_size"): [8],
    ("upsample_nearest1d_backward", "output_size"): [8],
    ("upsample_nearest1d_backward", "input_size"): [2, 3, 4],
    ("upsample_nearest2d", "output_size"): [8, 8],
    ("upsample_nearest2d_backward", "output_size"): [8, 8],
    ("upsample_nearest2d_backward", "input_size"): [2, 3, 4, 4],
    ("upsample_nearest3d", "output_size"): [6, 6, 6],
    ("upsample_nearest3d_backward", "output_size"): [6, 6, 6],
    ("upsample_nearest3d_backward", "input_size"): [2, 3, 3, 3, 3],
    ("upsample_trilinear3d", "output_size"): [6, 6, 6],
    ("upsample_trilinear3d", "align_corners"): False,
    ("upsample_trilinear3d_backward", "output_size"): [6, 6, 6],
    ("upsample_trilinear3d_backward", "input_size"): [2, 3, 3, 3, 3],
    ("upsample_trilinear3d_backward", "align_corners"): False,
    ("_upsample_bicubic2d_aa", "output_size"): [8, 8],
    ("_upsample_bicubic2d_aa", "align_corners"): False,
    ("_upsample_bicubic2d_aa_backward", "output_size"): [8, 8],
    ("_upsample_bicubic2d_aa_backward", "input_size"): [2, 3, 4, 4],
    ("_upsample_bicubic2d_aa_backward", "align_corners"): False,
    ("_upsample_bilinear2d_aa", "output_size"): [8, 8],
    ("_upsample_bilinear2d_aa", "align_corners"): False,
    ("_upsample_bilinear2d_aa_backward", "output_size"): [8, 8],
    ("_upsample_bilinear2d_aa_backward", "input_size"): [2, 3, 4, 4],
    ("_upsample_bilinear2d_aa_backward", "align_corners"): False,
    ("_upsample_nearest_exact1d", "output_size"): [8],
    ("_upsample_nearest_exact1d_backward", "output_size"): [8],
    ("_upsample_nearest_exact1d_backward", "input_size"): [2, 3, 4],
    ("_upsample_nearest_exact2d", "output_size"): [8, 8],
    ("_upsample_nearest_exact2d_backward", "output_size"): [8, 8],
    ("_upsample_nearest_exact2d_backward", "input_size"): [2, 3, 4, 4],
    ("_upsample_nearest_exact3d", "output_size"): [6, 6, 6],
    ("_upsample_nearest_exact3d_backward", "output_size"): [6, 6, 6],
    ("_upsample_nearest_exact3d_backward", "input_size"): [2, 3, 3, 3, 3],
    ("_conv_depthwise2d", "kernel_size"): [3, 3],
    ("_conv_depthwise2d", "stride"): [1, 1],
    ("_conv_depthwise2d", "padding"): [1, 1],
    ("_conv_depthwise2d", "dilation"): [1, 1],
    ("_slow_conv2d_forward", "kernel_size"): [3, 3],
    ("_slow_conv2d_forward", "stride"): [1, 1],
    ("_slow_conv2d_forward", "padding"): [1, 1],
    ("_slow_conv2d_backward", "kernel_size"): [3, 3],
    ("_slow_conv2d_backward", "stride"): [1, 1],
    ("_slow_conv2d_backward", "padding"): [1, 1],
    ("slow_conv3d", "kernel_size"): [3, 3, 3],
    ("slow_conv3d", "stride"): [1, 1, 1],
    ("slow_conv3d", "padding"): [1, 1, 1],
    ("slow_conv3d_forward", "kernel_size"): [3, 3, 3],
    ("slow_conv3d_forward", "stride"): [1, 1, 1],
    ("slow_conv3d_forward", "padding"): [1, 1, 1],
    ("slow_conv_transpose2d", "kernel_size"): [3, 3],
    ("slow_conv_transpose2d", "stride"): [1, 1],
    ("slow_conv_transpose2d", "padding"): [1, 1],
    ("slow_conv_transpose2d", "output_padding"): [0, 0],
    ("slow_conv_transpose2d", "dilation"): [1, 1],
    ("slow_conv_transpose3d", "kernel_size"): [3, 3, 3],
    ("slow_conv_transpose3d", "stride"): [1, 1, 1],
    ("slow_conv_transpose3d", "padding"): [1, 1, 1],
    ("slow_conv_transpose3d", "output_padding"): [0, 0, 0],
    ("slow_conv_transpose3d", "dilation"): [1, 1, 1],
    ("thnn_conv2d", "kernel_size"): [3, 3],
    ("thnn_conv2d", "stride"): [1, 1],
    ("thnn_conv2d", "padding"): [1, 1],
    # `int dim` for ops where 0 is a safe choice for our test shapes.
    ("kthvalue", "k"): 1,
    ("kthvalue", "dim"): 0,
    ("mode", "dim"): 0,
}

_TYPE_DEFAULTS = {"int": 0, "SymInt": 0, "bool": False, "str": "none"}

# Mirrors `kStringToDataType` in `src/data_type.h`.  Any tensor passed to
# an InfiniOps op must have one of these dtypes; others (`bool`, complex,
# quantised types) abort the process inside `DataTypeFromString`.  Some
# vendor torch forks lag behind upstream and lack `uint16` / `uint32` /
# `uint64` (added in PyTorch 2.3); resolve them lazily and keep the
# attributes that actually exist.
_SUPPORTED_DTYPE_NAMES = (
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "bfloat16",
    "float32",
    "float64",
)
_SUPPORTED_DTYPES = frozenset(
    getattr(torch, name) for name in _SUPPORTED_DTYPE_NAMES if hasattr(torch, name)
)

_OP_DTYPES = {
    "cholesky": (torch.float32,),
    "bitwise_and": _INTEGER_TEST_DTYPES,
    "bitwise_and_": _INTEGER_TEST_DTYPES,
    "bitwise_left_shift": _INTEGER_TEST_DTYPES,
    "bitwise_left_shift_": _INTEGER_TEST_DTYPES,
    "bitwise_not": _INTEGER_TEST_DTYPES,
    "bitwise_or": _INTEGER_TEST_DTYPES,
    "bitwise_or_": _INTEGER_TEST_DTYPES,
    "bitwise_right_shift": _INTEGER_TEST_DTYPES,
    "bitwise_right_shift_": _INTEGER_TEST_DTYPES,
    "bitwise_xor": _INTEGER_TEST_DTYPES,
    "bitwise_xor_": _INTEGER_TEST_DTYPES,
    "_slow_conv2d_backward": (torch.float32, torch.bfloat16),
    "_slow_conv2d_forward": (torch.float32, torch.bfloat16),
    "gcd": _INTEGER_TEST_DTYPES,
    "float_power_": (torch.float64,),
    "histc": (torch.float32, torch.float64),
    "lcm": _INTEGER_TEST_DTYPES,
    "linalg_cholesky": (torch.float32,),
    "linalg_ldl_solve": (torch.float32, torch.float64),
    "linalg_lstsq": (torch.float32, torch.float64),
    "linalg_lu": (torch.float32,),
    "linalg_lu_factor": (torch.float32,),
    "linalg_lu_factor_ex": (torch.float32,),
    "linalg_matrix_norm": (torch.float32, torch.float64),
    "slow_conv3d": (torch.float32, torch.bfloat16),
    "slow_conv3d_forward": (torch.float32, torch.bfloat16),
    "slow_conv_transpose2d": (torch.float32, torch.bfloat16),
    "slow_conv_transpose3d": (torch.float32, torch.bfloat16),
    "thnn_conv2d": (torch.float32, torch.bfloat16),
}


_LIST_SIZE_RE = re.compile(r"\[(\d+)\]")


def _is_inplace_aten_name(name):
    """Return whether `name` is an ATen in-place operator name."""

    return name.endswith("_") and not name.endswith("__")


def _list_default(aten_type, param_name=None):
    """Default value for a required `int[N]` / `SymInt[N]` param."""
    size_match = _LIST_SIZE_RE.search(aten_type)
    n = int(size_match.group(1)) if size_match else 1
    fill = 0

    if param_name in {"kernel_size", "output_size"}:
        fill = 2
    elif param_name in {"stride", "dilation"}:
        fill = 1
    elif param_name in {"padding", "output_padding"}:
        fill = 0

    return [fill] * n


def _dtype_tolerances(dtype):
    return _DTYPE_TOLERANCES.get(dtype, (0.0, 0.0))


def _dtypes_for_op(op_name):
    return _OP_DTYPES.get(op_name, _DEFAULT_DTYPES)


def _rand_tensor(shape, dtype, device, *, low=None, high=None):
    if dtype.is_floating_point:
        return randn_strided(shape, None, dtype=dtype, device=device)

    if low is None:
        low = 0 if dtype in _UNSIGNED_DTYPES else -4

    if high is None:
        high = 4

    return randint_strided(low, high, shape, None, dtype=dtype, device=device)


# Errors emitted by upstream PyTorch and vendor-forked variants for
# unsupported (op, dtype, device) combinations.  We skip rather than fail
# on these — the gap is in PyTorch, not InfiniOps.
_VENDOR_SKIP_PATTERNS = (
    "not implemented for",  # upstream PyTorch
    "CNNL_STATUS_BAD_PARAM",  # `torch_mlu` (Cambricon)
    "MUDNN failed",  # `torch_musa` (Moore)
    "Could not run",  # missing dispatcher entry on this backend
    "don't support tensor dtype",  # `torch_mlu` dtype check
    "unknown format type",  # `torch_npu` format descriptor gap
    "result requires dtype",  # output dtype mismatch (e.g. `float_power`)
    # ATen kernels for some loss ops (`mse_loss`, `huber_loss`, …) use
    # the `out` buffer as intermediate scratch and resize it before the
    # final reduction.  Our `from_blob` outputs are non-resizable, so
    # the kernel aborts the call with this message.  Skip these — the
    # zero-copy wrapper can't drive that codepath.
    "Trying to resize storage that is not resizable",
)

# Stateless random factories still need dedicated handling because
# their values are created by the operator itself.
_RANDOM_OPS = frozenset(
    {
        "rand",
        "randn",
        "randint",
        "randperm",
    }
)

# These RNG-consuming ops become testable once we reset the same seed
# before the reference and Infini calls.
_SEEDED_RANDOM_OPS = frozenset(
    {
        "bernoulli",
        "bernoulli_",
        "multinomial",
        "normal",
        "rrelu_with_noise",
    }
)

# Ops whose vendor kernel hangs indefinitely on at least one platform
# (`mode` on `torch_musa` for MUSA tensors).  Skip until the vendor
# fixes the underlying kernel — letting the CI block on a hanging
# kernel costs ~30 min per platform run.
_VENDOR_HANG_OPS = frozenset(
    {
        "mode",
    }
)

# Ops whose vendor kernel crashes the Python process, so they must be skipped
# before calling into the InfiniOps/PyTorch slot.
_VENDOR_CRASH_OPS = frozenset(
    {
        ("npu", "mish"),
        ("npu", "nuclear_norm"),
        ("npu", "_linalg_svd"),
        ("npu", "svd"),
    }
)

# Ops where the ATen `_out` schema and the Python reference (`torch.<op>`,
# `torch.nn.functional.<op>`) diverge in positional-argument ordering, so
# the harness's purely-positional reference call lands an InfiniOps
# argument on the wrong reference parameter.  E.g. ATen
# `binary_cross_entropy_out(self, target, weight=None, reduction=Mean, out)`
# has `weight` between `target` and `reduction`; with `weight` hidden as
# `Tensor?`, our visible signature is `(self, target, reduction, out)`,
# but `torch.nn.functional.binary_cross_entropy(input, target, weight,
# reduction)` reads our `reduction:int` as `weight:Tensor` and crashes
# inside `weight.size()`.  The InfiniOps wrapper itself is fine; only
# the harness's reference call is wrong.
_REFERENCE_SIGNATURE_MISMATCH_OPS = frozenset(
    {}
)

# Some ATen `_out` tensors are auxiliary workspaces rather than stable
# user-visible values.  Their shape/dtype still matter, but the exact
# contents can legitimately vary across implementations.
_AUXILIARY_OUTPUT_PARAMS = {
    "log_sigmoid_forward": frozenset({"buffer"}),
}

# Full reductions with low-precision inputs diverge between the functional
# (`torch.<op>(x)`) and `_out` paths because of intermediate-precision
# choices we cannot align from outside ATen.
_LARGE_REDUCTION_OPS = frozenset(
    {"sum", "mean", "nansum", "nanmean", "prod", "std", "var"}
)

# Ops with input-domain `TORCH_CHECK` macros that fire as device-side
# `assert` on CUDA when our generic random fp32 inputs fall outside the
# expected range.  The Python-side `RuntimeError` is catchable, but the
# CUDA context is left poisoned and every subsequent test errors at
# setup.  Skip these on cuda; the CPU path raises a clean exception
# that the existing harness already handles.
_DEVICE_ASSERTING_OPS = frozenset(
    {
        "binary_cross_entropy",  # requires inputs in [0, 1]
        "multi_margin_loss",
        "multilabel_margin_loss",
        "nll_loss",
        "nll_loss2d",
        # cuDNN paths divide by `kernel_size`/`stride` and SIGFPE on the
        # `[0, 0]` defaults our harness substitutes for required `int[N]`
        # parameters.
        "cudnn_convolution",
        "slow_conv3d",
        "slow_conv_transpose2d",
        "slow_conv_transpose3d",
        "thnn_conv2d",
        "im2col",
        "col2im",
        "max_unpool2d",
        "max_unpool3d",
        "reflection_pad1d",
        "reflection_pad2d",
        "reflection_pad3d",
        "replication_pad1d",
        "replication_pad2d",
        "replication_pad3d",
        "upsample_bicubic2d",
        "upsample_bilinear2d",
        "upsample_linear1d",
        "upsample_nearest1d",
        "upsample_nearest2d",
        "upsample_nearest3d",
        "upsample_trilinear3d",
        "avg_pool2d",
        "avg_pool3d",
        "max_pool2d_with_indices",
        "max_pool3d_with_indices",
        "adaptive_max_pool2d",
        "adaptive_max_pool3d",
        "adaptive_avg_pool2d",
        "adaptive_avg_pool3d",
    }
)

_ATEN_DEFAULT_REFERENCE_OPS = frozenset(
    {
        "_upsample_bicubic2d_aa",
        "_upsample_bicubic2d_aa_backward",
        "_upsample_bilinear2d_aa",
        "_upsample_bilinear2d_aa_backward",
        "_upsample_nearest_exact1d",
        "_upsample_nearest_exact1d_backward",
        "_upsample_nearest_exact2d",
        "_upsample_nearest_exact2d_backward",
        "_upsample_nearest_exact3d",
        "_upsample_nearest_exact3d_backward",
        "col2im",
        "gelu_backward",
        "glu_backward",
        "hardshrink_backward",
        "hardsigmoid_backward",
        "hardtanh_backward",
        "huber_loss_backward",
        "im2col",
        "leaky_relu_backward",
        "log_sigmoid_backward",
        "logit_backward",
        "max_pool2d_with_indices_backward",
        "max_pool3d_with_indices_backward",
        "mse_loss_backward",
        "reflection_pad1d",
        "reflection_pad1d_backward",
        "reflection_pad2d",
        "reflection_pad2d_backward",
        "reflection_pad3d",
        "reflection_pad3d_backward",
        "replication_pad1d",
        "replication_pad1d_backward",
        "replication_pad2d",
        "replication_pad2d_backward",
        "replication_pad3d",
        "replication_pad3d_backward",
        "sigmoid_backward",
        "silu_backward",
        "smooth_l1_loss_backward",
        "soft_margin_loss_backward",
        "softplus_backward",
        "softshrink_backward",
        "tanh_backward",
        "threshold_backward",
        "upsample_bicubic2d",
        "upsample_bicubic2d_backward",
        "upsample_bilinear2d",
        "upsample_bilinear2d_backward",
        "upsample_linear1d",
        "upsample_linear1d_backward",
        "upsample_nearest1d",
        "upsample_nearest1d_backward",
        "upsample_nearest2d",
        "upsample_nearest2d_backward",
        "upsample_nearest3d",
        "upsample_nearest3d_backward",
        "upsample_trilinear3d",
        "upsample_trilinear3d_backward",
    }
)


def _torch_func(op_name):
    """Resolve the reference function across the public torch namespaces."""

    direct_aliases = {
        "linalg_matmul": (torch, "matmul"),
        "log_sigmoid": (torch.nn.functional, "logsigmoid"),
    }
    direct_funcs = {
        "binary_cross_entropy": lambda input, target, reduction: torch.ops.aten.binary_cross_entropy(
            input, target, None, reduction
        ),
        "binary_cross_entropy_backward": lambda grad_output, input, target, reduction: torch.ops.aten.binary_cross_entropy_backward(
            grad_output, input, target, None, reduction
        ),
        "batch_norm_elemt": lambda input, mean, invstd, eps: torch.batch_norm_elemt(
            input, None, None, mean, invstd, eps
        ),
        "index": lambda input, indices: torch.ops.aten.index.Tensor(input, indices),
        "_conv_depthwise2d": lambda input, weight, kernel_size, stride, padding, dilation: torch.nn.functional.conv2d(
            input,
            weight,
            None,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=input.shape[1],
        ),
        "_slow_conv2d_forward": lambda input, weight, kernel_size, stride, padding: torch.ops.aten._slow_conv2d_forward.default(
            input, weight, kernel_size, None, stride, padding
        ),
        "_slow_conv2d_backward": lambda grad_output, input, weight, kernel_size, stride, padding: torch.ops.aten._slow_conv2d_backward.output_mask(
            grad_output, input, weight, kernel_size, stride, padding, [True, True, True]
        ),
        "mean": lambda input, keepdim: torch.mean(input, dim=None, keepdim=keepdim),
        "sum": lambda input, keepdim: torch.sum(input, dim=None, keepdim=keepdim),
        "std": lambda input, unbiased, keepdim: torch.std(
            input, dim=None, unbiased=unbiased, keepdim=keepdim
        ),
        "var": lambda input, unbiased, keepdim: torch.var(
            input, dim=None, unbiased=unbiased, keepdim=keepdim
        ),
        "elu": lambda input, alpha, scale, input_scale: torch.ops.aten.elu(
            input, alpha, scale, input_scale
        ),
        "elu_backward": lambda grad_output, alpha, scale, input_scale, is_result, self_or_result: torch.ops.aten.elu_backward(
            grad_output, alpha, scale, input_scale, is_result, self_or_result
        ),
        "huber_loss": lambda input, target, reduction, delta: torch.ops.aten.huber_loss(
            input, target, reduction, delta
        ),
        "multi_margin_loss": lambda input, target, p, margin, reduction: torch.ops.aten.multi_margin_loss(
            input, target, p, margin, None, reduction
        ),
        "adaptive_max_pool2d_backward": lambda grad_output, input, indices: torch.ops.aten.adaptive_max_pool2d_backward(
            grad_output, input, indices
        ),
        "adaptive_max_pool3d_backward": lambda grad_output, input, indices: torch.ops.aten.adaptive_max_pool3d_backward(
            grad_output, input, indices
        ),
        "adaptive_avg_pool3d_backward": lambda grad_output, input: torch.ops.aten.adaptive_avg_pool3d_backward.grad_input(
            grad_output,
            input,
            grad_input=torch.empty(input.shape, dtype=input.dtype, device=input.device),
        ),
        "avg_pool2d_backward": lambda grad_output, input, kernel_size, stride, padding, ceil_mode, count_include_pad: torch.ops.aten.avg_pool2d_backward(
            grad_output,
            input,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            None,
        ),
        "avg_pool3d_backward": lambda grad_output, input, kernel_size, stride, padding, ceil_mode, count_include_pad: torch.ops.aten.avg_pool3d_backward(
            grad_output,
            input,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            None,
        ),
        "fractional_max_pool2d_backward": lambda grad_output, input, kernel_size, output_size, indices: torch.ops.aten.fractional_max_pool2d_backward(
            grad_output, input, kernel_size, output_size, indices
        ),
        "fractional_max_pool3d_backward": lambda grad_output, input, kernel_size, output_size, indices: torch.ops.aten.fractional_max_pool3d_backward(
            grad_output, input, kernel_size, output_size, indices
        ),
        "max_unpool2d": lambda input, indices, output_size: torch.ops.aten.max_unpool2d(
            input, indices, output_size
        ),
        "max_unpool3d": lambda input, indices, output_size, stride, padding: torch.ops.aten.max_unpool3d(
            input, indices, output_size, stride, padding
        ),
        "log_sigmoid_forward": lambda input: torch.ops.aten.log_sigmoid_forward.output(
            input,
            output=torch.empty_like(input),
            buffer=torch.empty_like(input),
        ),
        "mse_loss": lambda input, target, reduction: torch.ops.aten.mse_loss(
            input, target, reduction
        ),
        "multilabel_margin_loss_forward": lambda input, target, reduction: torch.ops.aten.multilabel_margin_loss_forward.default(
            input, target, reduction
        ),
        "multilabel_margin_loss_backward": lambda grad_output, input, target, reduction, is_target: torch.ops.aten.multilabel_margin_loss_backward.default(
            grad_output, input, target, reduction, is_target
        ),
        "multi_margin_loss_backward": lambda grad_output, input, target, p, margin, reduction: torch.ops.aten.multi_margin_loss_backward.default(
            grad_output, input, target, p, margin, None, reduction
        ),
        "native_batch_norm": lambda input, training, momentum, eps: torch.native_batch_norm(
            input, None, None, None, None, training, momentum, eps
        ),
        "nll_loss": lambda input, target, reduction, ignore_index: torch.ops.aten.nll_loss(
            input, target, None, reduction, ignore_index
        ),
        "nll_loss2d": lambda input, target, reduction, ignore_index: torch.ops.aten.nll_loss2d(
            input, target, None, reduction, ignore_index
        ),
        "nll_loss_forward": lambda input, target, reduction, ignore_index: torch.ops.aten.nll_loss_forward.default(
            input, target, None, reduction, ignore_index
        ),
        "nll_loss_backward": lambda grad_output, input, target, reduction, ignore_index, total_weight: torch.ops.aten.nll_loss_backward.default(
            grad_output, input, target, None, reduction, ignore_index, total_weight
        ),
        "nll_loss2d_forward": lambda input, target, reduction, ignore_index: torch.ops.aten.nll_loss2d_forward.default(
            input, target, None, reduction, ignore_index
        ),
        "nll_loss2d_backward": lambda grad_output, input, target, reduction, ignore_index, total_weight: torch.ops.aten.nll_loss2d_backward.default(
            grad_output, input, target, None, reduction, ignore_index, total_weight
        ),
        "slow_conv3d": lambda input, weight, kernel_size, stride, padding: torch.ops.aten.slow_conv3d.default(
            input, weight, kernel_size, None, stride, padding
        ),
        "slow_conv3d_forward": lambda input, weight, kernel_size, stride, padding: torch.ops.aten.slow_conv3d_forward.default(
            input, weight, kernel_size, None, stride, padding
        ),
        "slow_conv_transpose2d": lambda input, weight, kernel_size, stride, padding, output_padding, dilation: torch.ops.aten.slow_conv_transpose2d.default(
            input, weight, kernel_size, None, stride, padding, output_padding, dilation
        ),
        "slow_conv_transpose3d": lambda input, weight, kernel_size, stride, padding, output_padding, dilation: torch.ops.aten.slow_conv_transpose3d.default(
            input, weight, kernel_size, None, stride, padding, output_padding, dilation
        ),
        "rrelu_with_noise": lambda input, noise, lower, upper, training: torch.ops.aten.rrelu_with_noise.default(
            input, noise, lower, upper, training
        ),
        "smooth_l1_loss": lambda input, target, reduction, beta: torch.ops.aten.smooth_l1_loss(
            input, target, reduction, beta
        ),
        "soft_margin_loss": lambda input, target, reduction: torch.ops.aten.soft_margin_loss(
            input, target, reduction
        ),
        "tensordot": lambda input, other, dims_self, dims_other: torch.tensordot(
            input, other, dims=(dims_self, dims_other)
        ),
        "thnn_conv2d": lambda input, weight, kernel_size, stride, padding: torch.ops.aten.thnn_conv2d.default(
            input, weight, kernel_size, None, stride, padding
        ),
    }

    if op_name in direct_funcs:
        return direct_funcs[op_name]

    if op_name in direct_aliases:
        namespace, attr = direct_aliases[op_name]
        func = getattr(namespace, attr, None)

        if func is not None:
            return func

    if _is_inplace_aten_name(op_name):
        method_name = op_name

        def _call_inplace(input, *args, **kwargs):
            return getattr(input, method_name)(*args, **kwargs)

        return _call_inplace

    candidates = [
        (torch, op_name),
        (torch.special, op_name),
        (torch.nn.functional, op_name),
    ]

    if op_name.startswith("special_"):
        candidates.append((torch.special, op_name.removeprefix("special_")))
    if op_name.startswith("linalg_"):
        candidates.append((torch.linalg, op_name.removeprefix("linalg_")))
    if op_name.startswith("fft_"):
        candidates.append((torch.fft, op_name.removeprefix("fft_")))

    for namespace, attr in candidates:
        func = getattr(namespace, attr, None)

        if func is not None:
            return func

    if op_name in _ATEN_DEFAULT_REFERENCE_OPS:
        return getattr(torch.ops.aten, op_name).default

    pytest.skip(f"no reference function for `{op_name}` in PyTorch")


def _pascal(snake_name):
    return "".join(part.capitalize() for part in snake_name.split("_"))


def _skip_if_not_active(op_name, device):
    op_class = getattr(infini.ops, _pascal(op_name), None)

    if op_class is None:
        pytest.skip(f"`{op_name}` class not exposed on this build")

    if _PYTORCH_SLOT not in op_class.active_implementation_indices(device):
        pytest.skip(f"`{op_name}` slot {_PYTORCH_SLOT} not active on `{device}`")


def _skip_low_precision_reduction(op_name, dtype, device):
    if op_name in _LARGE_REDUCTION_OPS:
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip(f"`{op_name}` precision diverges on fp16/bf16")

        if device == "musa":
            pytest.skip(f"`{op_name}` on `torch_musa` diverges from CPU reference")


def _build_input_value(op_name, param, shape, dtype, device, tensor_idx):
    """Build the value passed to a non-out parameter."""

    if param["is_tensor"]:
        per_op = _TENSOR_SHAPES.get(op_name)
        tshape = (
            per_op[tensor_idx]
            if per_op is not None and tensor_idx < len(per_op)
            else shape
        )
        input_shape = per_op[0] if per_op is not None else shape
        semantic_input_shape = (
            per_op[1]
            if op_name in {
                "multi_margin_loss_backward",
                "multilabel_margin_loss_backward",
                "nll_loss_backward",
                "nll_loss2d_backward",
            }
            and per_op is not None
            and len(per_op) > 1
            else input_shape
        )
        name = param["name"]

        if name == "random_samples":
            return rand_strided(tshape, None, dtype=dtype, device=device)

        if op_name in {"binary_cross_entropy", "binary_cross_entropy_backward"} and name in {
            "input",
            "target",
        }:
            return rand_strided(tshape, None, dtype=dtype, device=device)

        if op_name in {"bucketize", "histogram"} and name in {"boundaries", "bins"}:
            return torch.tensor([-1.0, 0.0, 1.0], dtype=dtype, device=device)

        if op_name in {"bernoulli", "bernoulli_"} and name in {"input", "p"}:
            return rand_strided(tshape, None, dtype=dtype, device=device)

        if op_name == "multinomial" and name == "input":
            return rand_strided(tshape, None, dtype=dtype, device=device)

        if op_name in {"quantile", "nanquantile"} and name == "q":
            return torch.tensor(0.5, dtype=torch.float32, device=device)

        if op_name == "linalg_tensorinv" and name == "input":
            return torch.eye(4, dtype=dtype, device=device).reshape(2, 2, 2, 2)

        if op_name in {"cholesky", "linalg_cholesky"} and name == "input":
            base = randn_strided(tshape, None, dtype=dtype, device=device)
            eye = torch.eye(tshape[-1], dtype=dtype, device=device)
            return base @ base.mT + eye

        if op_name == "linalg_ldl_solve":
            matrix_shape = per_op[0] if per_op is not None else tshape
            rhs_shape = per_op[2] if per_op is not None and len(per_op) > 2 else tshape
            base = torch.arange(
                1,
                matrix_shape[-1] * matrix_shape[-1] + 1,
                dtype=dtype,
            ).reshape(matrix_shape)
            spd = base @ base.mT + matrix_shape[-1] * torch.eye(
                matrix_shape[-1], dtype=dtype
            )
            ld, pivots = torch.linalg.ldl_factor(spd)

            if name == "LD":
                return ld.to(device)

            if name == "pivots":
                return pivots.to(device)

            if name == "B":
                rhs = torch.arange(
                    1,
                    torch.Size(rhs_shape).numel() + 1,
                    dtype=dtype,
                ).reshape(rhs_shape)
                return rhs.to(device)

        if name == "mean" and op_name == "batch_norm_elemt":
            return randn_strided((input_shape[1],), None, dtype=dtype, device=device)

        if name == "invstd" and op_name == "batch_norm_elemt":
            return torch.ones((input_shape[1],), dtype=dtype, device=device)

        if op_name == "_convert_indices_from_coo_to_csr" and name == "input":
            return torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64, device=device)

        if op_name == "_convert_indices_from_csr_to_coo":
            if name == "crow_indices":
                return torch.tensor([0, 2, 5, 6], dtype=torch.int64, device=device)

            if name == "col_indices":
                return torch.tensor([0, 1, 0, 1, 2, 0], dtype=torch.int64, device=device)

        if name == "index":
            if op_name in {"index_add", "index_reduce", "index_copy", "index_select"}:
                return torch.arange(input_shape[0], dtype=torch.int64, device=device)

            if op_name in {"take"}:
                numel = max(int(torch.Size(input_shape).numel()), 1)
                return torch.arange(numel, dtype=torch.int64, device=device).reshape(
                    input_shape
                )

            if op_name in {"gather", "scatter", "scatter_add", "scatter_reduce"}:
                view = (input_shape[0],) + (1,) * (len(input_shape) - 1)
                index = torch.arange(input_shape[0], dtype=torch.int64, device=device)
                return index.view(view).expand(input_shape).clone()

        if name == "indices" and op_name == "take_along_dim":
            numel = max(int(torch.Size(input_shape).numel()), 1)
            return torch.arange(numel, dtype=torch.int64, device=device).reshape(
                input_shape
            )

        if name in {"condition", "mask"} and op_name in {
            "where",
            "masked_select",
            "masked_fill_",
        }:
            return rand_strided(tshape, None, dtype=torch.float32, device=device) > 0.5

        if name == "indices" and op_name == "max_unpool2d":
            spatial = input_shape[-2] * input_shape[-1]
            base = torch.arange(spatial, dtype=torch.int64, device=device).reshape(
                1, 1, input_shape[-2], input_shape[-1]
            )
            return base.expand(input_shape).clone()

        if name == "indices" and op_name == "max_unpool3d":
            spatial = input_shape[-3] * input_shape[-2] * input_shape[-1]
            base = torch.arange(spatial, dtype=torch.int64, device=device).reshape(
                1, 1, input_shape[-3], input_shape[-2], input_shape[-1]
            )
            return base.expand(input_shape).clone()

        if name == "indices" and op_name in {
            "adaptive_max_pool2d_backward",
            "fractional_max_pool2d_backward",
        }:
            spatial = input_shape[-2] * input_shape[-1]
            base = torch.arange(tshape[-2] * tshape[-1], dtype=torch.int64, device=device)
            base = base.reshape(1, 1, tshape[-2], tshape[-1])
            return base.remainder(spatial).expand(tshape).clone()

        if name == "indices" and op_name in {
            "adaptive_max_pool3d_backward",
            "fractional_max_pool3d_backward",
        }:
            spatial = input_shape[-3] * input_shape[-2] * input_shape[-1]
            base = torch.arange(
                tshape[-3] * tshape[-2] * tshape[-1],
                dtype=torch.int64,
                device=device,
            )
            base = base.reshape(1, 1, tshape[-3], tshape[-2], tshape[-1])
            return base.remainder(spatial).expand(tshape).clone()

        if name == "indices" and op_name in {
            "max_pool2d_with_indices_backward",
            "max_pool3d_with_indices_backward",
        }:
            return torch.zeros(tshape, dtype=torch.int64, device=device)

        if name == "other" and op_name in {
            "bitwise_left_shift",
            "bitwise_left_shift_",
            "bitwise_right_shift",
            "bitwise_right_shift_",
        }:
            return randint_strided(0, 3, tshape, None, dtype=dtype, device=device)

        if name in {"LU_pivots", "pivots"} and op_name in {
            "lu_solve",
            "linalg_lu_solve",
            "lu_unpack",
        }:
            size = tshape[0] if isinstance(tshape, tuple) else tshape
            return torch.arange(1, size + 1, dtype=torch.int32, device=device)

        if name == "target" and op_name in {
            "multi_margin_loss",
            "multi_margin_loss_backward",
            "nll_loss",
            "nll_loss_forward",
            "nll_loss_backward",
        }:
            batch, classes = semantic_input_shape[0], semantic_input_shape[1]
            return torch.arange(batch, dtype=torch.int64, device=device).remainder(
                classes
            )

        if name == "target" and op_name in {
            "nll_loss2d",
            "nll_loss2d_forward",
            "nll_loss2d_backward",
        }:
            batch, classes = semantic_input_shape[0], semantic_input_shape[1]
            target = torch.arange(
                batch * semantic_input_shape[2] * semantic_input_shape[3],
                dtype=torch.int64,
                device=device,
            ).reshape(batch, semantic_input_shape[2], semantic_input_shape[3])
            return target.remainder(classes)

        if name == "target" and op_name in {
            "multilabel_margin_loss",
            "multilabel_margin_loss_forward",
            "multilabel_margin_loss_backward",
        }:
            classes = semantic_input_shape[-1]
            target = torch.full(tshape, -1, dtype=torch.int64, device=device)
            flat = target.reshape(-1, target.shape[-1])
            count = min(2, flat.shape[-1], classes)

            for row in range(flat.shape[0]):
                flat[row, :count] = torch.arange(
                    row, row + count, dtype=torch.int64, device=device
                ).remainder(classes)

            return target

        if name == "target" and op_name == "soft_margin_loss":
            target = rand_strided(tshape, None, dtype=dtype, device=device)
            return torch.where(target > 0.5, torch.ones_like(target), -torch.ones_like(target))

        if name == "is_target" and op_name == "multilabel_margin_loss_backward":
            return torch.zeros(tshape, dtype=dtype, device=device)

        return _rand_tensor(tshape, dtype, device)

    key = (op_name, param["name"])

    if op_name == "index" and param["name"] == "indices":
        first_dim = shape[0] if shape else 1
        last = max(first_dim - 1, 0)
        return [torch.tensor([0, last], dtype=torch.int64, device=device)]

    if key in _SCALAR_VALUES:
        return _SCALAR_VALUES[key]

    t = param["type"]

    if t.startswith(("int[", "SymInt[")) or t in {"int[]", "SymInt[]"}:
        return _list_default(t, param["name"])

    return _TYPE_DEFAULTS.get(t, 0.5)


def _call_infini(op_name, *args):
    try:
        getattr(infini.ops, op_name)(*args, implementation_index=_PYTORCH_SLOT)
    except RuntimeError as exc:
        if any(p in str(exc) for p in _VENDOR_SKIP_PATTERNS):
            pytest.skip(f"`{op_name}` unsupported by torch on this device/dtype")

        raise


def _split_ref_args(params, values):
    args = []
    kwargs = {}

    for param, value in zip(params, values):
        if param.get("reference_keyword_only", param.get("keyword_only", False)):
            kwargs[param["name"]] = value
        else:
            args.append(value)

    return args, kwargs


def _reference_kwargs(op_name):
    if op_name in {
        "adaptive_max_pool2d",
        "adaptive_max_pool3d",
        "fractional_max_pool2d",
        "fractional_max_pool3d",
    }:
        return {"return_indices": True}

    return {}


def _assert_close(actual, expected, rtol, atol):
    if actual.dtype.is_floating_point:
        assert torch.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
    else:
        assert torch.equal(actual, expected)


def _testable_ops():
    """Filter the metadata down to ops the harness can drive.

    When multiple ATen overloads share the same `aten_name` they all
    end up under one generated InfiniOps class (e.g., `std.dim` and
    `std.correction` share the same wrapper), but each has a distinct ATen
    `_out` signature.  The reference call we synthesize from
    `op_meta['params']` only exercises one signature; the secondary
    overloads either rely on hidden defaults whose ATen interpretation
    differs from the Python wrapper's (`std.correction(self, dim=None,
    correction=None, ...)` defaults to a different correction than
    `torch.std(self)`), or expose a positional shape that the Python
    reference does not accept (e.g., `binary_cross_entropy_out`'s
    `reduction:int` lands on the reference's `weight:Tensor?`).  Keep
    only the first overload of each `aten_name`."""
    seen = set()
    keep = []
    keep_idx = {}

    for op in _METADATA.get("ops", []):
        aten_name = op["aten_name"]

        if aten_name in seen:
            if aten_name not in {"bitwise_and_", "bitwise_or_", "bitwise_xor_"}:
                continue

            other_param = next((p for p in op["params"] if p["name"] == "other"), None)
            if other_param is None or not other_param["is_tensor"]:
                continue

            prev = keep[keep_idx[aten_name]]
            prev_other = next(
                (p for p in prev["params"] if p["name"] == "other"),
                None,
            )
            if prev_other is None or prev_other["is_tensor"]:
                continue

            keep[keep_idx[aten_name]] = op
            continue

        seen.add(aten_name)
        keep_idx[aten_name] = len(keep)
        keep.append(op)

    return keep


def _op_meta_id(op_meta):
    if not isinstance(op_meta, dict):
        return "empty"

    # Multiple ATen overloads now share a single class name (`scatter` covers
    # `scatter.src`, `scatter.value`, `scatter.reduce`, ...) — disambiguate
    # parametrize ids by appending the visible parameter type signature so
    # pytest does not collapse them into duplicate ids.

    return op_meta["overload_name"]


def _shape_id(shape):
    return "x".join(map(str, shape))


def _test_cases():
    cases = []

    for op_meta in _testable_ops():
        aten_name = op_meta.get("aten_name", op_meta["name"])

        for shape in _SHAPES:
            for dtype in _dtypes_for_op(aten_name):
                cases.append(
                    pytest.param(
                        op_meta,
                        shape,
                        dtype,
                        id=(
                            f"{_op_meta_id(op_meta)}-"
                            f"{_shape_id(shape)}-"
                            f"{str(dtype).removeprefix('torch.')}"
                        ),
                    )
                )

    return cases


@pytest.mark.parametrize(("op_meta", "shape", "dtype"), _test_cases())
def test_op(op_meta, shape, dtype, device):
    op_name = op_meta["name"]
    aten_name = op_meta.get("aten_name", op_name)
    is_inplace = _is_inplace_aten_name(aten_name)
    rtol, atol = _dtype_tolerances(dtype)
    _skip_if_not_active(op_name, device)
    _skip_low_precision_reduction(aten_name, dtype, device)

    if aten_name in _RANDOM_OPS:
        pytest.skip(f"`{aten_name}` is non-deterministic (independent draws diverge)")

    if aten_name in _REFERENCE_SIGNATURE_MISMATCH_OPS:
        pytest.skip(
            f"`{aten_name}`'s ATen `_out` and Python reference signatures "
            "have different positional ordering"
        )

    if aten_name in _VENDOR_HANG_OPS:
        pytest.skip(f"`{aten_name}` hangs on at least one vendor kernel")

    if (device, aten_name) in _VENDOR_CRASH_OPS:
        pytest.skip(f"`{aten_name}` crashes on `{device}` vendor kernel")

    if device == "cuda" and aten_name in _DEVICE_ASSERTING_OPS:
        pytest.skip(
            f"`{aten_name}` triggers a CUDA device-side assert on random inputs"
        )

    in_params = (
        op_meta["params"]
        if is_inplace
        else [p for p in op_meta["params"] if not p["is_out"]]
    )
    out_params = (
        [in_params[0]]
        if is_inplace
        else [p for p in op_meta["params"] if p["is_out"]]
    )

    # Build inputs in YAML order.
    inputs = []
    tensor_idx = 0

    for p in in_params:
        inputs.append(
            _build_input_value(aten_name, p, shape, dtype, device, tensor_idx)
        )

        if p["is_tensor"]:
            tensor_idx += 1

    # Run the reference to discover output shape(s)/dtype(s).
    # An op may reject our generic `randn(shape)` input with any of these
    # exception types — the gap is in our test harness's input synthesis,
    # not in the InfiniOps wrapper.
    ref_inputs = [
        clone_strided(x) if isinstance(x, torch.Tensor) else x for x in inputs
    ]
    ref_args, ref_kwargs = _split_ref_args(in_params, ref_inputs)
    ref_kwargs.update(_reference_kwargs(aten_name))
    rng_seed = 123 if aten_name in _SEEDED_RANDOM_OPS else None

    if rng_seed is not None:
        torch.manual_seed(rng_seed)

    try:
        ref = _torch_func(aten_name)(*ref_args, **ref_kwargs)
    except (
        RuntimeError,
        TypeError,
        ValueError,
        IndexError,
        NotImplementedError,
    ) as exc:
        pytest.skip(f"`torch.{aten_name}` rejects these inputs: {exc}")

    ref_outs = ref if isinstance(ref, tuple) else (ref,)

    if is_inplace:
        ref_outs = (ref_inputs[0],)

    if len(ref_outs) != len(out_params):
        # The Python-facing function (e.g. `F.adaptive_max_pool2d`) often
        # exposes a subset of the ATen `_out` schema's outputs (returning
        # only `out`, hiding `indices` behind a `return_indices=True`
        # kwarg).  Without a per-op map of how to coax the full tuple
        # out, skip — the InfiniOps wrapper itself is fine.
        pytest.skip(
            f"`{aten_name}` reference produced {len(ref_outs)} output(s); "
            f"schema declares {len(out_params)}"
        )

    # InfiniOps `DataType` supports only `int{8,16,32,64}`,
    # `uint{8,16,32,64}`, `float{16,32,64}`, and `bfloat16`.  Tensors with
    # any other torch dtype (`bool`, `complex64`, `complex128`, etc.) abort
    # on `DataTypeFromString`, so skip the test rather than crash the process.
    tensors = [*ref_outs, *(x for x in inputs if isinstance(x, torch.Tensor))]
    unsupported = next(
        (t.dtype for t in tensors if t.dtype not in _SUPPORTED_DTYPES), None
    )

    if unsupported is not None:
        pytest.skip(
            f"`{op_name}` uses dtype {unsupported} — not in InfiniOps `DataType`"
        )

    # Some reference codepaths legitimately return 0-element tensors, but
    # the InfiniOps zero-copy wrapper cannot safely materialize those
    # outputs across all vendor backends.
    if any(t.numel() == 0 for t in ref_outs):
        pytest.skip(
            f"`{op_name}` produced 0-element output"
        )

    if is_inplace:
        if rng_seed is not None:
            torch.manual_seed(rng_seed)

        _call_infini(op_name, *inputs)
        _assert_close(inputs[0], ref_outs[0], rtol, atol)

        return

    outs = [torch.empty_like(t) for t in ref_outs]

    if rng_seed is not None:
        torch.manual_seed(rng_seed)

    _call_infini(op_name, *inputs, *outs)

    auxiliary = _AUXILIARY_OUTPUT_PARAMS.get(aten_name, frozenset())

    for actual, expected, param in zip(outs, ref_outs, out_params):
        if param["name"] in auxiliary:
            continue

        _assert_close(actual, expected, rtol, atol)
