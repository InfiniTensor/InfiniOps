# RMSNorm 用例设计文档

## 1. 算子标杆

PyTorch 参考实现：
```python
import torch

def rms_norm_ref(input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """CPU 参考实现，使用 float32 精度计算。"""
    input_fp32 = input.float()
    variance = input_fp32.pow(2).mean(dim=-1, keepdim=True)
    hidden_states = input_fp32 * torch.rsqrt(variance + eps)
    return (hidden_states * weight.float()).to(input.dtype)
```

NPU 调用方式（ascend_kernel 工程算子）：
```python
import torch
import ascend_kernel

# input: [*, hidden_dim], weight: [hidden_dim]
output = ascend_kernel.ops.rms_norm(input.npu(), weight.npu(), eps)
```

---

## 2. 用例说明

### 2.1 测试配置

```python
# 支持的数据类型
SUPPORTED_DTYPES = [torch.float16, torch.float32]

# 典型用例 — 模型常见 hidden_dim + batch 组合
TEST_SHAPES = [
    # (category, description, input_shape, hidden_dim_is_last_dim)
    ("2D", "small 32x128",               (32, 128)),
    ("2D", "medium 64x512",              (64, 512)),
    ("2D", "medium 128x1024",            (128, 1024)),
    ("2D", "Qwen/Llama 32x4096",         (32, 4096)),
    ("2D", "Qwen/Llama 128x4096",        (128, 4096)),
    ("2D", "Llama-70B 32x8192",          (32, 8192)),
    ("3D", "multi-head 4x32x128",        (4, 32, 128)),
    ("3D", "multi-head 8x64x512",        (8, 64, 512)),
    ("3D", "batch 4x128x4096",           (4, 128, 4096)),
]

# 泛化用例 — 边界和大规模场景
GENERAL_SHAPES = [
    # 小 shape 场景（边界测试）
    ("Small", "single row",              (1, 128)),
    ("Small", "single row 4096",         (1, 4096)),
    ("Small", "two rows",                (2, 256)),
    ("Small", "tiny 3D",                 (1, 1, 128)),
    ("Small", "non-aligned rows 3",      (3, 512)),
    ("Small", "non-aligned rows 7",      (7, 1024)),

    # 大 shape 场景（生产环境）
    ("Large", "BERT-base 512x768",       (512, 768)),
    ("Large", "GPT-2 1024x1024",         (1024, 1024)),
    ("Large", "Llama batch 256x4096",    (256, 4096)),
    ("Large", "Llama-70B batch 64x8192", (64, 8192)),
    ("Large", "3D large 8x512x4096",    (8, 512, 4096)),
]

# 边界值测试 — eps 和特殊输入
BOUNDARY_VALUES = [
    ("eps_small",   "very small eps",     (32, 512),  {"eps": 1e-12}),
    ("eps_large",   "large eps",          (32, 512),  {"eps": 1e-2}),
    ("zeros",       "all-zero input",     (16, 1024), {"input_fill": 0.0}),
    ("ones",        "all-one input",      (16, 1024), {"input_fill": 1.0}),
    ("large_vals",  "large input values", (16, 1024), {"input_scale": 100.0}),
    ("small_vals",  "tiny input values",  (16, 1024), {"input_scale": 1e-4}),
]
```

### 2.2 用例覆盖统计

| 类别 | Shape 数量 | 边界值数量 | dtype 数量 | 总用例数 |
|------|-----------|-----------|-----------|---------|
| 常规形状 (TEST_SHAPES) | 9 | — | 2 | 18 |
| 泛化形状 (GENERAL_SHAPES) | 11 | — | 2 | 22 |
| 边界值 (BOUNDARY_VALUES) | — | 6 | 2 | 12 |
| **总计** | **20** | **6** | **2** | **52** |

---

## 3. 使用说明

### 生成测试数据示例

```python
import torch

def generate_rms_norm_inputs(shape, dtype, eps=1e-6, input_fill=None, input_scale=1.0):
    """生成 rms_norm 测试输入。"""
    hidden_dim = shape[-1]
    weight = torch.randn(hidden_dim, dtype=dtype)

    if input_fill is not None:
        input_tensor = torch.full(shape, input_fill, dtype=dtype)
    else:
        input_tensor = torch.randn(shape, dtype=dtype) * input_scale

    expected = rms_norm_ref(input_tensor, weight, eps)

    return input_tensor, weight, eps, expected
```

### 注意事项

1. **weight shape**：始终为 `[hidden_dim]`（1D），`hidden_dim = input.shape[-1]`。
2. **eps 类型**：Python `float`（double），Host 端转 `float` 传给 kernel。
3. **fp16 精度**：参考实现中先升精度到 float32 计算，结果再降回 float16。测试对比时应考虑 fp16 的精度损失（rtol=1e-3, atol=1e-3）。
4. **全零输入**：`rsqrt(0 + eps)` 应正常工作，不应产生 nan/inf。
