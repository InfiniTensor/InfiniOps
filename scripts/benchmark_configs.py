"""Operator shape/dtype configurations and FLOPS calculations for benchmarking.

Provides LLM-inference-relevant shapes for all native InfiniOps operators and
key PyTorch fallback operators, along with FLOPS estimation functions.
"""

import re
import torch

# ---------------------------------------------------------------------------
# LLM model geometry constants (LLaMA-7B / 70B)
# ---------------------------------------------------------------------------
# LLaMA-7B
H_7B = 4096          # hidden_size
FFN_7B = 11008       # intermediate_size
N_HEADS_7B = 32      # num_attention_heads
HEAD_DIM_7B = 128    # head_dim = H / N_HEADS
KV_HEADS_7B = 32     # num_key_value_heads (MHA)

# LLaMA-70B
H_70B = 8192
FFN_70B = 28672
N_HEADS_70B = 64
HEAD_DIM_70B = 128
KV_HEADS_70B = 8     # GQA

# Sequence lengths
SEQ_DECODE = 1
SEQ_PREFILL_SHORT = 512
SEQ_PREFILL_LONG = 2048

# Batch sizes
BS_DECODE = 1
BS_PREFILL = 8

# ---------------------------------------------------------------------------
# Native operator shape configs: {op_name: {mode: [(shape_args, description)]}}
# ---------------------------------------------------------------------------
# shape_args varies per op: tensors + scalar params needed for setup.
# description is a human-readable label.
# ---------------------------------------------------------------------------

NATIVE_OP_SHAPES = {
    "add": {
        "quick": [
            ((BS_DECODE, H_7B), "LLaMA-7B decode residual add"),
        ],
        "standard": [
            ((BS_DECODE, H_7B), "LLaMA-7B decode residual add"),
            ((BS_PREFILL, H_7B), "LLaMA-7B prefill residual add"),
            ((BS_DECODE, H_70B), "LLaMA-70B decode residual add"),
        ],
        "thorough": [
            ((BS_DECODE, H_7B), "LLaMA-7B decode, bs=1"),
            ((4, H_7B), "LLaMA-7B decode, bs=4"),
            ((BS_PREFILL, H_7B), "LLaMA-7B prefill, bs=8"),
            ((16, H_7B), "LLaMA-7B prefill, bs=16"),
            ((BS_DECODE, H_70B), "LLaMA-70B decode, bs=1"),
            ((4, H_70B), "LLaMA-70B decode, bs=4"),
            ((BS_PREFILL, H_70B), "LLaMA-70B prefill, bs=8"),
        ],
    },
    "mul": {
        "quick": [
            ((BS_DECODE, H_7B), "LLaMA-7B decode element-wise mul"),
        ],
        "standard": [
            ((BS_DECODE, H_7B), "LLaMA-7B decode element-wise mul"),
            ((BS_PREFILL, H_7B), "LLaMA-7B prefill element-wise mul"),
        ],
        "thorough": [
            ((BS_DECODE, H_7B), "LLaMA-7B decode, bs=1"),
            ((4, H_7B), "LLaMA-7B decode, bs=4"),
            ((BS_PREFILL, H_7B), "LLaMA-7B prefill, bs=8"),
            ((16, H_7B), "LLaMA-7B prefill, bs=16"),
            ((BS_DECODE, H_70B), "LLaMA-70B decode, bs=1"),
            ((BS_PREFILL, H_70B), "LLaMA-70B prefill, bs=8"),
        ],
    },
    "cast": {
        "quick": [
            ((BS_DECODE, H_7B), "LLaMA-7B decode cast"),
        ],
        "standard": [
            ((BS_DECODE, H_7B), "LLaMA-7B decode cast"),
            ((BS_PREFILL, H_7B), "LLaMA-7B prefill cast"),
        ],
        "thorough": [
            ((BS_DECODE, H_7B), "LLaMA-7B decode, bs=1"),
            ((BS_PREFILL, H_7B), "LLaMA-7B prefill, bs=8"),
            ((BS_DECODE, H_70B), "LLaMA-70B decode, bs=1"),
            ((BS_PREFILL, H_70B), "LLaMA-70B prefill, bs=8"),
        ],
    },
    "cat": {
        "quick": [
            (((BS_DECODE, H_7B // 2), (BS_DECODE, H_7B // 2), 1), "LLaMA-7B decode cat dim=1"),
        ],
        "standard": [
            (((BS_DECODE, H_7B // 2), (BS_DECODE, H_7B // 2), 1), "LLaMA-7B decode cat dim=1"),
            (((BS_PREFILL, H_7B // 2), (BS_PREFILL, H_7B // 2), 1), "LLaMA-7B prefill cat dim=1"),
        ],
        "thorough": [
            (((BS_DECODE, H_7B // 2), (BS_DECODE, H_7B // 2), 1), "LLaMA-7B decode cat, 2 tensors"),
            (((BS_DECODE, H_7B // 4),) * 4 + (1,), "LLaMA-7B decode cat, 4 tensors"),
            (((BS_PREFILL, H_7B // 2), (BS_PREFILL, H_7B // 2), 1), "LLaMA-7B prefill cat, 2 tensors"),
        ],
    },
    "gemm": {
        "quick": [
            ((BS_DECODE, H_7B), (H_7B, FFN_7B), "LLaMA-7B decode FFN gate"),
        ],
        "standard": [
            ((BS_DECODE, H_7B), (H_7B, FFN_7B), "LLaMA-7B decode FFN gate"),
            ((BS_PREFILL, H_7B), (H_7B, FFN_7B), "LLaMA-7B prefill FFN gate"),
            ((BS_DECODE, H_70B), (H_70B, FFN_70B), "LLaMA-70B decode FFN gate"),
        ],
        "thorough": [
            ((BS_DECODE, H_7B), (H_7B, FFN_7B), "LLaMA-7B decode FFN gate"),
            ((BS_DECODE, H_7B), (H_7B, H_7B), "LLaMA-7B decode QKV proj"),
            ((BS_PREFILL, H_7B), (H_7B, FFN_7B), "LLaMA-7B prefill FFN gate"),
            ((16, H_7B), (H_7B, FFN_7B), "LLaMA-7B prefill bs=16 FFN gate"),
            ((BS_DECODE, H_70B), (H_70B, FFN_70B), "LLaMA-70B decode FFN gate"),
            ((BS_DECODE, H_70B), (H_70B, H_70B), "LLaMA-70B decode QKV proj"),
            ((BS_PREFILL, H_70B), (H_70B, FFN_70B), "LLaMA-70B prefill FFN gate"),
        ],
    },
    "matmul": {
        "quick": [
            ((BS_DECODE, H_7B), (H_7B, FFN_7B), "LLaMA-7B decode matmul"),
        ],
        "standard": [
            ((BS_DECODE, H_7B), (H_7B, FFN_7B), "LLaMA-7B decode matmul"),
            ((BS_PREFILL, H_7B), (H_7B, FFN_7B), "LLaMA-7B prefill matmul"),
            ((BS_DECODE, H_70B), (H_70B, FFN_70B), "LLaMA-70B decode matmul"),
        ],
        "thorough": [
            ((BS_DECODE, H_7B), (H_7B, FFN_7B), "LLaMA-7B decode matmul"),
            ((BS_PREFILL, H_7B), (H_7B, FFN_7B), "LLaMA-7B prefill matmul"),
            ((BS_DECODE, H_70B), (H_70B, FFN_70B), "LLaMA-70B decode matmul"),
            ((BS_PREFILL, H_70B), (H_70B, FFN_70B), "LLaMA-70B prefill matmul"),
        ],
    },
    "linear": {
        "quick": [
            ((BS_DECODE, H_7B), (H_7B, FFN_7B), "LLaMA-7B decode linear"),
        ],
        "standard": [
            ((BS_DECODE, H_7B), (H_7B, FFN_7B), "LLaMA-7B decode linear"),
            ((BS_PREFILL, H_7B), (H_7B, FFN_7B), "LLaMA-7B prefill linear"),
            ((BS_DECODE, H_70B), (H_70B, FFN_70B), "LLaMA-70B decode linear"),
        ],
        "thorough": [
            ((BS_DECODE, H_7B), (H_7B, FFN_7B), "LLaMA-7B decode linear"),
            ((BS_PREFILL, H_7B), (H_7B, FFN_7B), "LLaMA-7B prefill linear"),
            ((BS_DECODE, H_70B), (H_70B, FFN_70B), "LLaMA-70B decode linear"),
            ((BS_PREFILL, H_70B), (H_70B, FFN_70B), "LLaMA-70B prefill linear"),
        ],
    },
    "rms_norm": {
        "quick": [
            ((BS_DECODE, H_7B), "LLaMA-7B decode RMS norm"),
        ],
        "standard": [
            ((BS_DECODE, H_7B), "LLaMA-7B decode RMS norm"),
            ((BS_PREFILL, H_7B), "LLaMA-7B prefill RMS norm"),
            ((BS_DECODE, H_70B), "LLaMA-70B decode RMS norm"),
        ],
        "thorough": [
            ((BS_DECODE, H_7B), "LLaMA-7B decode RMS norm"),
            ((BS_PREFILL, H_7B), "LLaMA-7B prefill RMS norm"),
            ((BS_DECODE, H_70B), "LLaMA-70B decode RMS norm"),
            ((BS_PREFILL, H_70B), "LLaMA-70B prefill RMS norm"),
        ],
    },
    "causal_softmax": {
        "quick": [
            ((BS_DECODE, N_HEADS_7B, SEQ_DECODE), "LLaMA-7B decode causal softmax"),
        ],
        "standard": [
            ((BS_DECODE, N_HEADS_7B, SEQ_DECODE), "LLaMA-7B decode causal softmax"),
            ((BS_PREFILL, N_HEADS_7B, SEQ_PREFILL_SHORT), "LLaMA-7B prefill causal softmax"),
            ((BS_DECODE, N_HEADS_70B, SEQ_DECODE), "LLaMA-70B decode causal softmax"),
        ],
        "thorough": [
            ((BS_DECODE, N_HEADS_7B, SEQ_DECODE), "LLaMA-7B decode causal softmax"),
            ((BS_PREFILL, N_HEADS_7B, SEQ_PREFILL_SHORT), "LLaMA-7B prefill causal softmax S=512"),
            ((BS_PREFILL, N_HEADS_7B, SEQ_PREFILL_LONG), "LLaMA-7B prefill causal softmax S=2048"),
            ((BS_DECODE, N_HEADS_70B, SEQ_DECODE), "LLaMA-70B decode causal softmax"),
            ((BS_PREFILL, N_HEADS_70B, SEQ_PREFILL_SHORT), "LLaMA-70B prefill causal softmax"),
        ],
    },
    "swiglu": {
        "quick": [
            ((BS_DECODE, FFN_7B), "LLaMA-7B decode SwiGLU"),
        ],
        "standard": [
            ((BS_DECODE, FFN_7B), "LLaMA-7B decode SwiGLU"),
            ((BS_PREFILL, FFN_7B), "LLaMA-7B prefill SwiGLU"),
            ((BS_DECODE, FFN_70B), "LLaMA-70B decode SwiGLU"),
        ],
        "thorough": [
            ((BS_DECODE, FFN_7B), "LLaMA-7B decode SwiGLU"),
            ((BS_PREFILL, FFN_7B), "LLaMA-7B prefill SwiGLU"),
            ((BS_DECODE, FFN_70B), "LLaMA-70B decode SwiGLU"),
            ((BS_PREFILL, FFN_70B), "LLaMA-70B prefill SwiGLU"),
        ],
    },
    "flash_attention": {
        "quick": [
            ((BS_DECODE, N_HEADS_7B, HEAD_DIM_7B),
             (BS_DECODE, KV_HEADS_7B, HEAD_DIM_7B),
             SEQ_DECODE, "LLaMA-7B decode flash attn"),
        ],
        "standard": [
            ((BS_DECODE, N_HEADS_7B, HEAD_DIM_7B),
             (BS_DECODE, KV_HEADS_7B, HEAD_DIM_7B),
             SEQ_DECODE, "LLaMA-7B decode flash attn"),
            ((BS_PREFILL, N_HEADS_7B, HEAD_DIM_7B),
             (BS_PREFILL, KV_HEADS_7B, HEAD_DIM_7B),
             SEQ_PREFILL_SHORT, "LLaMA-7B prefill flash attn S=512"),
            ((BS_DECODE, N_HEADS_70B, HEAD_DIM_70B),
             (BS_DECODE, KV_HEADS_70B, HEAD_DIM_70B),
             SEQ_DECODE, "LLaMA-70B decode flash attn (GQA)"),
        ],
        "thorough": [
            ((BS_DECODE, N_HEADS_7B, HEAD_DIM_7B),
             (BS_DECODE, KV_HEADS_7B, HEAD_DIM_7B),
             SEQ_DECODE, "LLaMA-7B decode flash attn"),
            ((BS_PREFILL, N_HEADS_7B, HEAD_DIM_7B),
             (BS_PREFILL, KV_HEADS_7B, HEAD_DIM_7B),
             SEQ_PREFILL_SHORT, "LLaMA-7B prefill flash attn S=512"),
            ((BS_PREFILL, N_HEADS_7B, HEAD_DIM_7B),
             (BS_PREFILL, KV_HEADS_7B, HEAD_DIM_7B),
             SEQ_PREFILL_LONG, "LLaMA-7B prefill flash attn S=2048"),
            ((BS_DECODE, N_HEADS_70B, HEAD_DIM_70B),
             (BS_DECODE, KV_HEADS_70B, HEAD_DIM_70B),
             SEQ_DECODE, "LLaMA-70B decode flash attn (GQA)"),
            ((BS_PREFILL, N_HEADS_70B, HEAD_DIM_70B),
             (BS_PREFILL, KV_HEADS_70B, HEAD_DIM_70B),
             SEQ_PREFILL_SHORT, "LLaMA-70B prefill flash attn (GQA)"),
        ],
    },
    "rotary_embedding": {
        "quick": [
            ((BS_DECODE, N_HEADS_7B, HEAD_DIM_7B),
             (BS_DECODE, KV_HEADS_7B, HEAD_DIM_7B),
             SEQ_DECODE, HEAD_DIM_7B, "LLaMA-7B decode RoPE"),
        ],
        "standard": [
            ((BS_DECODE, N_HEADS_7B, HEAD_DIM_7B),
             (BS_DECODE, KV_HEADS_7B, HEAD_DIM_7B),
             SEQ_DECODE, HEAD_DIM_7B, "LLaMA-7B decode RoPE"),
            ((BS_PREFILL, N_HEADS_7B, HEAD_DIM_7B),
             (BS_PREFILL, KV_HEADS_7B, HEAD_DIM_7B),
             SEQ_PREFILL_SHORT, HEAD_DIM_7B, "LLaMA-7B prefill RoPE"),
        ],
        "thorough": [
            ((BS_DECODE, N_HEADS_7B, HEAD_DIM_7B),
             (BS_DECODE, KV_HEADS_7B, HEAD_DIM_7B),
             SEQ_DECODE, HEAD_DIM_7B, "LLaMA-7B decode RoPE"),
            ((BS_PREFILL, N_HEADS_7B, HEAD_DIM_7B),
             (BS_PREFILL, KV_HEADS_7B, HEAD_DIM_7B),
             SEQ_PREFILL_SHORT, HEAD_DIM_7B, "LLaMA-7B prefill RoPE"),
            ((BS_DECODE, N_HEADS_70B, HEAD_DIM_70B),
             (BS_DECODE, KV_HEADS_70B, HEAD_DIM_70B),
             SEQ_DECODE, HEAD_DIM_70B, "LLaMA-70B decode RoPE"),
        ],
    },
    "add_rms_norm": {
        "quick": [
            ((BS_DECODE, H_7B), "LLaMA-7B decode add+RMS norm"),
        ],
        "standard": [
            ((BS_DECODE, H_7B), "LLaMA-7B decode add+RMS norm"),
            ((BS_PREFILL, H_7B), "LLaMA-7B prefill add+RMS norm"),
        ],
        "thorough": [
            ((BS_DECODE, H_7B), "LLaMA-7B decode add+RMS norm"),
            ((BS_PREFILL, H_7B), "LLaMA-7B prefill add+RMS norm"),
            ((BS_DECODE, H_70B), "LLaMA-70B decode add+RMS norm"),
            ((BS_PREFILL, H_70B), "LLaMA-70B prefill add+RMS norm"),
        ],
    },
    "reshape_and_cache": {
        "quick": [
            ((BS_DECODE, KV_HEADS_7B, HEAD_DIM_7B), "LLaMA-7B decode KV cache write"),
        ],
        "standard": [
            ((BS_DECODE, KV_HEADS_7B, HEAD_DIM_7B), "LLaMA-7B decode KV cache write"),
            ((BS_PREFILL, KV_HEADS_7B, HEAD_DIM_7B), "LLaMA-7B prefill KV cache write"),
            ((BS_DECODE, KV_HEADS_70B, HEAD_DIM_70B), "LLaMA-70B decode KV cache write (GQA)"),
        ],
        "thorough": [
            ((BS_DECODE, KV_HEADS_7B, HEAD_DIM_7B), "LLaMA-7B decode KV cache write"),
            ((BS_PREFILL, KV_HEADS_7B, HEAD_DIM_7B), "LLaMA-7B prefill KV cache write"),
            ((BS_DECODE, KV_HEADS_70B, HEAD_DIM_70B), "LLaMA-70B decode KV cache write (GQA)"),
            ((BS_PREFILL, KV_HEADS_70B, HEAD_DIM_70B), "LLaMA-70B prefill KV cache write (GQA)"),
        ],
    },
}

# ---------------------------------------------------------------------------
# Fallback operator shape configs: {category: {op_name: {mode: shapes}}}
# ---------------------------------------------------------------------------

# Matrix operations need compatible shapes
_FALLBACK_MATRIX_SHAPES = {
    "mm": {
        "quick": [((BS_DECODE, H_7B), (H_7B, FFN_7B))],
        "standard": [
            ((BS_DECODE, H_7B), (H_7B, FFN_7B)),
            ((BS_PREFILL, H_7B), (H_7B, FFN_7B)),
        ],
        "thorough": [
            ((BS_DECODE, H_7B), (H_7B, FFN_7B)),
            ((BS_PREFILL, H_7B), (H_7B, FFN_7B)),
            ((BS_DECODE, H_70B), (H_70B, FFN_70B)),
            ((BS_PREFILL, H_70B), (H_70B, FFN_70B)),
        ],
    },
    "bmm": {
        "quick": [((BS_DECODE, N_HEADS_7B, HEAD_DIM_7B), (BS_DECODE, HEAD_DIM_7B, SEQ_DECODE))],
        "standard": [
            ((BS_DECODE, N_HEADS_7B, HEAD_DIM_7B), (BS_DECODE, HEAD_DIM_7B, SEQ_DECODE)),
            ((BS_PREFILL, N_HEADS_7B, HEAD_DIM_7B), (BS_PREFILL, HEAD_DIM_7B, SEQ_PREFILL_SHORT)),
        ],
        "thorough": [
            ((BS_DECODE, N_HEADS_7B, HEAD_DIM_7B), (BS_DECODE, HEAD_DIM_7B, SEQ_DECODE)),
            ((BS_PREFILL, N_HEADS_7B, HEAD_DIM_7B), (BS_PREFILL, HEAD_DIM_7B, SEQ_PREFILL_SHORT)),
        ],
    },
    "addmm": {
        "quick": [((BS_DECODE, H_7B), (H_7B, FFN_7B))],
        "standard": [
            ((BS_DECODE, H_7B), (H_7B, FFN_7B)),
            ((BS_PREFILL, H_7B), (H_7B, FFN_7B)),
        ],
        "thorough": [
            ((BS_DECODE, H_7B), (H_7B, FFN_7B)),
            ((BS_PREFILL, H_7B), (H_7B, FFN_7B)),
            ((BS_DECODE, H_70B), (H_70B, FFN_70B)),
        ],
    },
    "baddbmm": {
        "quick": [((BS_DECODE, N_HEADS_7B, HEAD_DIM_7B), (BS_DECODE, HEAD_DIM_7B, SEQ_DECODE))],
        "standard": [
            ((BS_DECODE, N_HEADS_7B, HEAD_DIM_7B), (BS_DECODE, HEAD_DIM_7B, SEQ_DECODE)),
        ],
        "thorough": [
            ((BS_DECODE, N_HEADS_7B, HEAD_DIM_7B), (BS_DECODE, HEAD_DIM_7B, SEQ_DECODE)),
            ((BS_PREFILL, N_HEADS_7B, HEAD_DIM_7B), (BS_PREFILL, HEAD_DIM_7B, SEQ_PREFILL_SHORT)),
        ],
    },
}

# Elementwise unary shapes (reused for sigmoid, silu, gelu, relu, tanh, etc.)
_FALLBACK_ELEMENTWISE_UNARY_SHAPES = {
    "quick": [(BS_DECODE, H_7B)],
    "standard": [(BS_DECODE, H_7B), (BS_PREFILL, H_7B), (BS_DECODE, H_70B)],
    "thorough": [
        (BS_DECODE, H_7B), (4, H_7B), (BS_PREFILL, H_7B),
        (BS_DECODE, H_70B), (BS_PREFILL, H_70B),
    ],
}

# Elementwise binary shapes (same-shape)
_FALLBACK_ELEMENTWISE_BINARY_SHAPES = {
    "quick": [(BS_DECODE, H_7B)],
    "standard": [(BS_DECODE, H_7B), (BS_PREFILL, H_7B), (BS_DECODE, H_70B)],
    "thorough": [
        (BS_DECODE, H_7B), (BS_PREFILL, H_7B),
        (BS_DECODE, H_70B), (BS_PREFILL, H_70B),
    ],
}

# Reduction shapes
_FALLBACK_REDUCTION_SHAPES = {
    "quick": [(BS_DECODE, H_7B)],
    "standard": [(BS_DECODE, H_7B), (BS_PREFILL, H_7B), (BS_DECODE, H_70B)],
    "thorough": [
        (BS_DECODE, H_7B), (BS_PREFILL, H_7B),
        (BS_DECODE, H_70B), (BS_PREFILL, H_70B),
    ],
}

# Fallback op categories — maps category to list of (op_name, shape_config)
FALLBACK_OP_CONFIGS = {
    "matrix": {
        "ops": ["mm", "bmm", "addmm", "baddbmm"],
        "shapes": _FALLBACK_MATRIX_SHAPES,
    },
    "activation": {
        "ops": ["sigmoid", "silu", "gelu", "threshold", "tanh", "relu",
                "hardtanh", "elu", "softplus"],
        "shapes": _FALLBACK_ELEMENTWISE_UNARY_SHAPES,
    },
    "normalization": {
        "ops": ["softmax", "log_softmax"],
        "shapes": _FALLBACK_ELEMENTWISE_UNARY_SHAPES,
    },
    "reduction": {
        "ops": ["sum", "mean", "amax", "amin", "argmax", "argmin", "norm",
                "prod", "var", "std"],
        "shapes": _FALLBACK_REDUCTION_SHAPES,
    },
    "elementwise": {
        "ops": ["abs", "neg", "sqrt", "exp", "rsqrt", "reciprocal", "pow",
                "clamp", "clone", "copy"],
        "shapes": _FALLBACK_ELEMENTWISE_UNARY_SHAPES,
    },
    "comparison": {
        "ops": ["eq", "ne", "lt", "le", "gt", "ge", "maximum", "minimum"],
        "shapes": _FALLBACK_ELEMENTWISE_BINARY_SHAPES,
    },
    "other": {
        "ops": ["where", "gather", "scatter", "index_select", "topk", "sort"],
        "shapes": _FALLBACK_ELEMENTWISE_UNARY_SHAPES,
    },
}

# ---------------------------------------------------------------------------
# Per-op scalar parameter defaults (from test_torch_ops.py)
# ---------------------------------------------------------------------------
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
    ("index_reduce", "reduce"): "amax",
    ("scatter_reduce", "reduce"): "amax",
    ("scatter_reduce_two", "reduce"): "amax",
    ("kthvalue_values", "k"): 1,
    ("kthvalue_values", "dim"): 0,
    ("mode_values", "dim"): 0,
    ("clamp", "min"): -1.0,
    ("clamp", "max"): 1.0,
    ("softmax", "dim"): -1,
    ("gelu", "approximate"): "none",
    ("sub", "alpha"): 1.0,
    ("addmm", "beta"): 1.0,
    ("addmm", "alpha"): 1.0,
    ("avg_pool2d", "kernel_size"): [3, 3],
    ("avg_pool2d", "stride"): [1, 1],
    ("avg_pool2d", "padding"): [1, 1],
    ("avg_pool2d", "ceil_mode"): False,
    ("avg_pool2d", "count_include_pad"): True,
    ("topk", "k"): 5,
    ("sort", "dim"): -1,
    ("gather", "dim"): 0,
    ("scatter", "dim"): 0,
    ("index_select", "dim"): 0,
    ("norm", "p"): 2.0,
    ("pow", "exponent"): 2.0,
    ("where", "condition"): None,  # will be built dynamically
    ("sum", "dim"): -1,
    ("mean", "dim"): -1,
    ("amax", "dim"): -1,
    ("amin", "dim"): -1,
    ("argmax", "dim"): -1,
    ("argmin", "dim"): -1,
    ("var", "dim"): -1,
    ("std", "dim"): -1,
}

_TYPE_DEFAULTS = {"int": 0, "SymInt": 0, "bool": False, "str": "none"}

_LIST_SIZE_RE = re.compile(r"\[(\d+)\]")

# ---------------------------------------------------------------------------
# Skip patterns for fallback ops
# ---------------------------------------------------------------------------
_RANDOM_OPS = frozenset({
    "bernoulli", "bernoulli_", "multinomial", "normal", "rand", "randn",
    "randint", "randperm", "rrelu_with_noise",
})

_DEVICE_ASSERTING_OPS = frozenset({
    "binary_cross_entropy", "multi_margin_loss", "multilabel_margin_loss",
    "nll_loss", "nll_loss2d", "cudnn_convolution", "slow_conv3d",
    "slow_conv_transpose2d", "slow_conv_transpose3d", "thnn_conv2d",
    "im2col", "col2im", "max_unpool2d", "max_unpool3d",
    "reflection_pad1d", "reflection_pad2d", "reflection_pad3d",
    "replication_pad1d", "replication_pad2d", "replication_pad3d",
    "upsample_bicubic2d", "upsample_bilinear2d", "upsample_linear1d",
    "upsample_nearest1d", "upsample_nearest2d", "upsample_nearest3d",
    "upsample_trilinear3d", "avg_pool2d", "avg_pool3d",
    "max_pool2d_with_indices", "max_pool3d_with_indices",
    "adaptive_max_pool2d", "adaptive_max_pool3d",
    "adaptive_avg_pool2d", "adaptive_avg_pool3d",
})

_VENDOR_SKIP_PATTERNS = (
    "not implemented for",
    "CNNL_STATUS_BAD_PARAM",
    "MUDNN failed",
    "Could not run",
    "don't support tensor dtype",
    "unknown format type",
    "result requires dtype",
    "Trying to resize storage that is not resizable",
)

# Ops with special tensor shape requirements (from test_torch_ops.py)
_TENSOR_SHAPES = {
    "mm": ((8, 16), (16, 12)),
    "bmm": ((4, 8, 16), (4, 16, 12)),
    "matmul": ((8, 16), (16, 12)),
    "dot": ((16,), (16,)),
    "vdot": ((16,), (16,)),
    "mv": ((8, 16), (16,)),
    "inner": ((8, 16), (8, 16)),
    "outer": ((8,), (12,)),
    "ger": ((8,), (12,)),
    "kron": ((3, 4), (2, 3)),
}


# ---------------------------------------------------------------------------
# FLOPS / throughput estimation
# ---------------------------------------------------------------------------

def compute_elementwise_flops(num_elements):
    """FLOPS for a single elementwise operation (1 FLOP per element)."""
    return num_elements


def compute_gemm_flops(M, N, K):
    """FLOPS for a GEMM operation: 2*M*N*K."""
    return 2 * M * N * K


def compute_rms_norm_flops(num_elements):
    """FLOPS for RMS norm: x^2 + mean + sqrt + div + mul = ~5 FLOP/element."""
    return 5 * num_elements


def compute_softmax_flops(num_elements):
    """FLOPS for softmax: exp + sum + div = ~3 FLOP/element."""
    return 3 * num_elements


def compute_swiglu_flops(num_elements):
    """FLOPS for SwiGLU: sigmoid + mul + mul = ~4 FLOP/element."""
    return 4 * num_elements


def compute_flash_attention_flops(T, H, D, S):
    """FLOPS for flash attention: Q*K^T + attn*V.

    Q*K^T: 2 * T * H * D * S
    attn*V: 2 * T * H * S * D
    Total:  4 * T * H * D * S
    """
    return 4 * T * H * D * S


def compute_rotary_embedding_flops(T, H, D):
    """FLOPS for rotary embedding: ~6 FLOP per (token, head, dim) pair.

    cos/sin lookup + complex multiply = ~6 FLOPs.
    """
    return 6 * T * H * D


def compute_add_rms_norm_flops(num_elements):
    """FLOPS for fused add + RMS norm: add(1) + rms_norm(5) = 6 FLOP/element."""
    return 6 * num_elements


def compute_reshape_and_cache_flops(num_tokens, num_kv_heads, head_size):
    """FLOPS for reshape_and_cache: essentially a scatter write, ~2 FLOP/element."""
    return 2 * num_tokens * num_kv_heads * head_size


def compute_data_volume_gb(tensors_or_shape, dtype):
    """Compute data volume in GB for a shape or list of tensors.

    Args:
        tensors_or_shape: a shape tuple, a list of shape tuples, or a torch.Tensor
        dtype: torch dtype
    """
    element_size = torch.tensor([], dtype=dtype).element_size()

    if isinstance(tensors_or_shape, torch.Tensor):
        return tensors_or_shape.numel() * element_size / 1e9

    if isinstance(tensors_or_shape[0], (list, tuple)):
        total = sum(s[0] * s[1] if len(s) == 2
                    else s[0] * s[1] * s[2] if len(s) == 3
                    else 1
                    for s in tensors_or_shape)
        return total * element_size / 1e9

    shape = tensors_or_shape
    numel = 1
    for d in shape:
        numel *= d
    return numel * element_size / 1e9


def num_elements(shape):
    """Total number of elements in a shape tuple."""
    n = 1
    for d in shape:
        n *= d
    return n


def shape_to_str(shape):
    """Convert a shape to a compact string representation."""
    if isinstance(shape, (list, tuple)):
        if len(shape) > 0 and isinstance(shape[0], (list, tuple)):
            return "x".join(
                "[" + ",".join(str(d) for d in s) + "]" for s in shape
            )
        return "[" + ",".join(str(d) for d in shape) + "]"
    return str(shape)


def get_scalar_default(op_name, param):
    """Get the default scalar value for a non-tensor parameter."""
    key = (op_name, param["name"])
    if key in _SCALAR_VALUES:
        return _SCALAR_VALUES[key]
    t = param["type"]
    if t.startswith(("int[", "SymInt[")) or t in {"int[]", "SymInt[]"}:
        size_match = _LIST_SIZE_RE.search(t)
        n = int(size_match.group(1)) if size_match else 1
        return [0] * n
    return _TYPE_DEFAULTS.get(t, 0.5)


# ---------------------------------------------------------------------------
# ntops operator configurations
# ---------------------------------------------------------------------------
# Each entry: op_name -> {type, torch_ref, extra_kwargs?}
# type: "unary" | "binary" | "mm" | "bmm" | "matmul" | "addmm" |
#       "rms_norm" | "layer_norm" | "softmax" | "clamp" | "dropout" |
#       "sdpa" | "rope" | "conv2d" | "max_pool2d" | "avg_pool2d"

NTOPS_OPS = {
    # -- Unary elementwise --
    "abs":           {"type": "unary"},
    "neg":           {"type": "unary"},
    "exp":           {"type": "unary"},
    "rsqrt":         {"type": "unary"},
    "sigmoid":       {"type": "unary"},
    "silu":          {"type": "unary"},
    "gelu":          {"type": "unary"},
    "relu":          {"type": "unary"},
    "tanh":          {"type": "unary"},
    "sin":           {"type": "unary"},
    "cos":           {"type": "unary"},
    "isinf":         {"type": "unary"},
    "isnan":         {"type": "unary"},
    "bitwise_not":   {"type": "unary"},
    "softmax":       {"type": "softmax"},
    # -- Binary elementwise --
    "add":           {"type": "binary"},
    "sub":           {"type": "binary"},
    "mul":           {"type": "binary"},
    "div":           {"type": "binary"},
    "pow":           {"type": "binary"},
    "eq":            {"type": "binary"},
    "ne":            {"type": "binary"},
    "lt":            {"type": "binary"},
    "le":            {"type": "binary"},
    "gt":            {"type": "binary"},
    "ge":            {"type": "binary"},
    "bitwise_and":   {"type": "binary"},
    "bitwise_or":    {"type": "binary"},
    # -- Matrix --
    "mm":            {"type": "mm"},
    "bmm":           {"type": "bmm"},
    "matmul":        {"type": "matmul"},
    "addmm":         {"type": "addmm"},
    # -- Norm --
    "rms_norm":      {"type": "rms_norm"},
    "layer_norm":    {"type": "layer_norm"},
    # -- Attention --
    "scaled_dot_product_attention": {"type": "sdpa"},
    "rotary_position_embedding":    {"type": "rope"},
    # -- Special --
    "clamp":         {"type": "clamp"},
    "dropout":       {"type": "dropout"},
    "conv2d":        {"type": "conv2d"},
    "max_pool2d":    {"type": "max_pool2d"},
    "avg_pool2d":    {"type": "avg_pool2d"},
}

# PyTorch reference function for each ntops op
NTOPS_TORCH_REF = {
    "abs":     lambda x: torch.abs(x),
    "neg":     lambda x: torch.neg(x),
    "exp":     lambda x: torch.exp(x),
    "rsqrt":   lambda x: torch.rsqrt(x),
    "sigmoid": lambda x: torch.sigmoid(x),
    "silu":    lambda x: torch.nn.functional.silu(x),
    "gelu":    lambda x: torch.nn.functional.gelu(x),
    "relu":    lambda x: torch.nn.functional.relu(x),
    "tanh":    lambda x: torch.tanh(x),
    "sin":     lambda x: torch.sin(x),
    "cos":     lambda x: torch.cos(x),
    "isinf":   lambda x: torch.isinf(x),
    "isnan":   lambda x: torch.isnan(x),
    "bitwise_not": lambda x: torch.bitwise_not(x),
    "softmax": lambda x: torch.nn.functional.softmax(x.float(), dim=-1).to(x.dtype),
    "add":     lambda x, y: torch.add(x, y),
    "sub":     lambda x, y: torch.sub(x, y),
    "mul":     lambda x, y: torch.mul(x, y),
    "div":     lambda x, y: torch.div(x, y),
    "pow":     lambda x, y: torch.pow(x, y),
    "eq":      lambda x, y: torch.eq(x, y),
    "ne":      lambda x, y: torch.ne(x, y),
    "lt":      lambda x, y: torch.lt(x, y),
    "le":      lambda x, y: torch.le(x, y),
    "gt":      lambda x, y: torch.gt(x, y),
    "ge":      lambda x, y: torch.ge(x, y),
    "bitwise_and": lambda x, y: torch.bitwise_and(x, y),
    "bitwise_or":  lambda x, y: torch.bitwise_or(x, y),
    "mm":      lambda a, b: torch.mm(a, b),
    "bmm":     lambda a, b: torch.bmm(a, b),
    "matmul":  lambda a, b: torch.matmul(a, b),
    "addmm":   lambda a, b, c: torch.addmm(c, a, b),
    "rms_norm": None,  # handled specially
    "layer_norm": None,  # handled specially
    "scaled_dot_product_attention": None,  # handled specially
    "rotary_position_embedding": None,  # handled specially
    "clamp":   lambda x: torch.clamp(x, min=-1.0, max=1.0),
    "dropout": lambda x: torch.nn.functional.dropout(x, p=0.1, training=True),
    "conv2d":  None,  # handled specially
    "max_pool2d": None,  # handled specially
    "avg_pool2d": None,  # handled specially
}

# ntops shape configurations per mode
NTOPS_UNARY_SHAPES = {
    "quick": [(BS_DECODE, H_7B)],
    "standard": [(BS_DECODE, H_7B), (BS_PREFILL, H_7B), (BS_DECODE, H_70B)],
    "thorough": [
        (BS_DECODE, H_7B), (4, H_7B), (BS_PREFILL, H_7B),
        (BS_DECODE, H_70B), (BS_PREFILL, H_70B),
    ],
}

NTOPS_BINARY_SHAPES = NTOPS_UNARY_SHAPES

NTOPS_MM_SHAPES = {
    "quick": [((BS_DECODE, H_7B), (H_7B, FFN_7B))],
    "standard": [
        ((BS_DECODE, H_7B), (H_7B, FFN_7B)),
        ((BS_PREFILL, H_7B), (H_7B, FFN_7B)),
    ],
    "thorough": [
        ((BS_DECODE, H_7B), (H_7B, FFN_7B)),
        ((BS_PREFILL, H_7B), (H_7B, FFN_7B)),
        ((BS_DECODE, H_70B), (H_70B, FFN_70B)),
    ],
}

NTOPS_BMM_SHAPES = {
    "quick": [((BS_DECODE, N_HEADS_7B, HEAD_DIM_7B), (BS_DECODE, HEAD_DIM_7B, SEQ_DECODE))],
    "standard": [
        ((BS_DECODE, N_HEADS_7B, HEAD_DIM_7B), (BS_DECODE, HEAD_DIM_7B, SEQ_DECODE)),
        ((BS_PREFILL, N_HEADS_7B, HEAD_DIM_7B), (BS_PREFILL, HEAD_DIM_7B, SEQ_PREFILL_SHORT)),
    ],
    "thorough": [
        ((BS_DECODE, N_HEADS_7B, HEAD_DIM_7B), (BS_DECODE, HEAD_DIM_7B, SEQ_DECODE)),
        ((BS_PREFILL, N_HEADS_7B, HEAD_DIM_7B), (BS_PREFILL, HEAD_DIM_7B, SEQ_PREFILL_SHORT)),
    ],
}

NTOPS_ADDMM_SHAPES = {
    "quick": [((BS_DECODE, H_7B), (H_7B, FFN_7B))],
    "standard": [
        ((BS_DECODE, H_7B), (H_7B, FFN_7B)),
        ((BS_PREFILL, H_7B), (H_7B, FFN_7B)),
    ],
    "thorough": [
        ((BS_DECODE, H_7B), (H_7B, FFN_7B)),
        ((BS_PREFILL, H_7B), (H_7B, FFN_7B)),
        ((BS_DECODE, H_70B), (H_70B, FFN_70B)),
    ],
}

NTOPS_SDPA_SHAPES = {
    "quick": [((BS_DECODE, N_HEADS_7B, SEQ_DECODE, HEAD_DIM_7B),
               (BS_DECODE, KV_HEADS_7B, SEQ_DECODE, HEAD_DIM_7B),
               (BS_DECODE, KV_HEADS_7B, SEQ_DECODE, HEAD_DIM_7B))],
    "standard": [
        ((BS_DECODE, N_HEADS_7B, SEQ_DECODE, HEAD_DIM_7B),
         (BS_DECODE, KV_HEADS_7B, SEQ_DECODE, HEAD_DIM_7B),
         (BS_DECODE, KV_HEADS_7B, SEQ_DECODE, HEAD_DIM_7B)),
        ((BS_PREFILL, N_HEADS_7B, SEQ_PREFILL_SHORT, HEAD_DIM_7B),
         (BS_PREFILL, KV_HEADS_7B, SEQ_PREFILL_SHORT, HEAD_DIM_7B),
         (BS_PREFILL, KV_HEADS_7B, SEQ_PREFILL_SHORT, HEAD_DIM_7B)),
    ],
    "thorough": [
        ((BS_DECODE, N_HEADS_7B, SEQ_DECODE, HEAD_DIM_7B),
         (BS_DECODE, KV_HEADS_7B, SEQ_DECODE, HEAD_DIM_7B),
         (BS_DECODE, KV_HEADS_7B, SEQ_DECODE, HEAD_DIM_7B)),
        ((BS_PREFILL, N_HEADS_7B, SEQ_PREFILL_SHORT, HEAD_DIM_7B),
         (BS_PREFILL, KV_HEADS_7B, SEQ_PREFILL_SHORT, HEAD_DIM_7B),
         (BS_PREFILL, KV_HEADS_7B, SEQ_PREFILL_SHORT, HEAD_DIM_7B)),
        ((BS_PREFILL, N_HEADS_7B, SEQ_PREFILL_LONG, HEAD_DIM_7B),
         (BS_PREFILL, KV_HEADS_7B, SEQ_PREFILL_LONG, HEAD_DIM_7B),
         (BS_PREFILL, KV_HEADS_7B, SEQ_PREFILL_LONG, HEAD_DIM_7B)),
    ],
}

NTOPS_ROPE_SHAPES = {
    "quick": [(BS_DECODE, N_HEADS_7B, HEAD_DIM_7B)],
    "standard": [
        (BS_DECODE, N_HEADS_7B, HEAD_DIM_7B),
        (BS_PREFILL, N_HEADS_7B, HEAD_DIM_7B),
    ],
    "thorough": [
        (BS_DECODE, N_HEADS_7B, HEAD_DIM_7B),
        (BS_PREFILL, N_HEADS_7B, HEAD_DIM_7B),
        (BS_DECODE, N_HEADS_70B, HEAD_DIM_70B),
    ],
}

# Map op name -> shape config (same dict interface as NTOPS_UNARY_SHAPES)
NTOPS_SHAPES = {}  # filled lazily per op type

# Conv2d / pool2d shapes (ResNet-like)
NTOPS_CONV2D_SHAPES = {
    "quick": [((1, 64, 56, 56), (64, 64, 3, 3))],
    "standard": [
        ((1, 64, 56, 56), (64, 64, 3, 3)),
        ((1, 128, 28, 28), (128, 128, 3, 3)),
    ],
    "thorough": [
        ((1, 64, 56, 56), (64, 64, 3, 3)),
        ((1, 128, 28, 28), (128, 128, 3, 3)),
        ((1, 256, 14, 14), (256, 256, 3, 3)),
    ],
}

NTOPS_POOL2D_SHAPES = {
    "quick": [(1, 64, 56, 56)],
    "standard": [(1, 64, 56, 56), (1, 128, 28, 28)],
    "thorough": [(1, 64, 56, 56), (1, 128, 28, 28), (1, 256, 14, 14)],
}


def get_ntops_shapes(op_name, mode):
    """Return list of shape configs for a ntops operator."""
    op_type = NTOPS_OPS[op_name]["type"]

    if op_type == "unary":
        return NTOPS_UNARY_SHAPES.get(mode, NTOPS_UNARY_SHAPES["quick"])
    elif op_type == "binary":
        return NTOPS_BINARY_SHAPES.get(mode, NTOPS_BINARY_SHAPES["quick"])
    elif op_type == "softmax":
        return NTOPS_UNARY_SHAPES.get(mode, NTOPS_UNARY_SHAPES["quick"])
    elif op_type == "mm":
        return NTOPS_MM_SHAPES.get(mode, NTOPS_MM_SHAPES["quick"])
    elif op_type == "bmm":
        return NTOPS_BMM_SHAPES.get(mode, NTOPS_BMM_SHAPES["quick"])
    elif op_type == "matmul":
        return NTOPS_MM_SHAPES.get(mode, NTOPS_MM_SHAPES["quick"])
    elif op_type == "addmm":
        return NTOPS_ADDMM_SHAPES.get(mode, NTOPS_ADDMM_SHAPES["quick"])
    elif op_type == "rms_norm":
        return NTOPS_UNARY_SHAPES.get(mode, NTOPS_UNARY_SHAPES["quick"])
    elif op_type == "layer_norm":
        return NTOPS_UNARY_SHAPES.get(mode, NTOPS_UNARY_SHAPES["quick"])
    elif op_type == "sdpa":
        return NTOPS_SDPA_SHAPES.get(mode, NTOPS_SDPA_SHAPES["quick"])
    elif op_type == "rope":
        return NTOPS_ROPE_SHAPES.get(mode, NTOPS_ROPE_SHAPES["quick"])
    elif op_type == "clamp":
        return NTOPS_UNARY_SHAPES.get(mode, NTOPS_UNARY_SHAPES["quick"])
    elif op_type == "dropout":
        return NTOPS_UNARY_SHAPES.get(mode, NTOPS_UNARY_SHAPES["quick"])
    elif op_type == "conv2d":
        return NTOPS_CONV2D_SHAPES.get(mode, NTOPS_CONV2D_SHAPES["quick"])
    elif op_type in ("max_pool2d", "avg_pool2d"):
        return NTOPS_POOL2D_SHAPES.get(mode, NTOPS_POOL2D_SHAPES["quick"])
    return NTOPS_UNARY_SHAPES.get(mode, NTOPS_UNARY_SHAPES["quick"])


# ---------------------------------------------------------------------------
# Fallback op FLOPS estimation
# ---------------------------------------------------------------------------

def estimate_fallback_flops(op_name, shapes, num_out_elements):
    """Estimate FLOPS for a fallback op. Returns None if unknown."""
    if op_name in _FALLBACK_MATRIX_SHAPES:
        if len(shapes) >= 2:
            a_shape, b_shape = shapes[0], shapes[1]
            M = num_elements(a_shape) // a_shape[-1]
            K = a_shape[-1]
            N = b_shape[-1]
            return compute_gemm_flops(M, N, K)
        return None

    # Elementwise unary
    unary_ops = {"sigmoid", "silu", "gelu", "threshold", "tanh", "relu",
                 "hardtanh", "elu", "softplus", "abs", "neg", "sqrt", "exp",
                 "rsqrt", "reciprocal", "clone", "copy"}
    if op_name in unary_ops:
        return compute_elementwise_flops(num_out_elements)

    if op_name in ("softmax", "log_softmax"):
        return compute_softmax_flops(num_out_elements)

    if op_name in ("pow", "clamp"):
        return 2 * num_out_elements

    # Binary elementwise
    binary_ops = {"eq", "ne", "lt", "le", "gt", "ge", "maximum", "minimum"}
    if op_name in binary_ops:
        return compute_elementwise_flops(num_out_elements)

    # Reductions
    if op_name in ("sum", "mean", "amax", "amin", "nansum", "nanmean", "prod"):
        return num_out_elements

    if op_name in ("argmax", "argmin"):
        return num_out_elements

    if op_name in ("norm", "var", "std"):
        return 2 * num_out_elements

    return None
