# `ascend_kernel.ops.rms_norm`

```python
torch.ops.npu.rms_norm(input, weight, eps=1e-6) → Tensor
```

对输入张量的最后一个维度执行 RMS 归一化（Root Mean Square Layer Normalization）。

$$y = x \cdot \frac{1}{\sqrt{\mathrm{mean}(x^2) + \varepsilon}} \cdot \text{weight}$$

与 LayerNorm 不同，RMSNorm 不减去均值，仅基于均方根进行归一化，计算开销更低。

## 参数说明

- **input** (`Tensor`) — 输入张量，维度 ≥ 1。归一化沿最后一个维度进行。
- **weight** (`Tensor`) — 一维权重张量，形状为 `[hidden_dim]`，其中 `hidden_dim = input.shape[-1]`。
- **eps** (`float`, 可选) — 加在方差上的小常数，防止除零。默认值 `1e-6`。

## 支持的数据类型

| 数据类型 | 支持 |
|---------|------|
| `torch.float16` | 是 |
| `torch.float32` | 是 |

`weight` 的数据类型可与 `input` 不同（内部统一转为 `float32` 计算）。

## Shape 约束

- `input`: 任意维度 ≥ 1 的张量，形状 `[*, hidden_dim]`。
- `weight`: 一维张量，形状 `[hidden_dim]`，必须满足 `weight.size(0) == input.size(-1)`。
- 输出与 `input` 同形状、同数据类型。

## 约束条件

- `hidden_dim`（对齐后）不能超过单核 UB 容量限制。在 Ascend 910B 上，`hidden_dim` 最大约 9600（`float32`）或 9600（`float16`）。
- `input` 和 `weight` 必须在 NPU 设备上。

## 使用示例

```python
import torch
import torch_npu
import ascend_kernel

# 基本用法。
x = torch.randn(32, 4096, dtype=torch.float16, device="npu")
w = torch.randn(4096, dtype=torch.float16, device="npu")
y = torch.ops.npu.rms_norm(x, w, 1e-6)

# 多维输入（batch × seq_len × hidden_dim）。
x = torch.randn(4, 128, 4096, dtype=torch.float32, device="npu")
w = torch.randn(4096, dtype=torch.float32, device="npu")
y = torch.ops.npu.rms_norm(x, w)  # eps 默认 1e-6
```

## 返回值

`Tensor` — 与 `input` 同形状、同数据类型的归一化结果。
