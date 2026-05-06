# RandomSample 算子接口设计文档

## 概述

将 InfiniCore 中的 `random_sample` 算子迁移到 InfiniOps，支持多硬件平台。

## 参考实现

- **InfiniCore**: `InfiniCore/src/infiniop/ops/random_sample/`
- **vLLM**: `vllm.v1.sample.sampler.Sampler`

---

## 接口设计

### 输入/输出张量

| 张量 | 形状 | 数据类型 | 说明 |
|------|------|----------|------|
| `logits` | `[batch_size, vocab_size]` | F16/BF16/F32 | 模型输出的原始 logits |
| `out` | `[batch_size]` | I32/I64 | 采样的 token indices |

### 采样参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `temperature` | `float` | `1.0` | 温度缩放，`< 1e-5` 视为 greedy |
| `top_k` | `int` | `0` | Top-k 过滤，`0` 表示不限制 |
| `top_p` | `float` | `1.0` | Top-p (nucleus) 过滤，`1.0` 表示不限制 |
| `random_val` | `float` | `0.0` | 随机种子值，`0` 表示 greedy (argmax) |

### 采样算法

```
1. 如果 temperature < 1e-5 或 random_val == 0:
       return argmax(logits)

2. Temperature Scaling:
       logits = logits / temperature

3. Softmax:
       probs = softmax(logits)

4. Top-k 过滤 (如果 top_k > 0):
       保留概率最大的 top_k 个 token

5. Top-p 过滤 (如果 top_p < 1.0):
       按概率降序排列，累积概率超过 top_p 的 token 被过滤

6. 采样:
       基于过滤后的概率分布，使用 random_val 进行 CDF 采样
```

---

## 基类代码草稿

```cpp
// src/base/random_sample.h

#ifndef INFINI_OPS_BASE_RANDOM_SAMPLE_H_
#define INFINI_OPS_BASE_RANDOM_SAMPLE_H_

#include <optional>

#include "operator.h"

namespace infini::ops {

class RandomSample : public Operator<RandomSample> {
 public:
  RandomSample(const Tensor logits, Tensor out,
               std::optional<float> temperature,
               std::optional<int> top_k,
               std::optional<float> top_p,
               std::optional<float> random_val)
      : temperature_{temperature.value_or(1.0f)},
        top_k_{top_k.value_or(0)},
        top_p_{top_p.value_or(1.0f)},
        random_val_{random_val.value_or(0.0f)},
        logits_dtype_{logits.dtype()},
        out_dtype_{out.dtype()},
        batch_size_{logits.size(-2)},
        vocab_size_{logits.size(-1)} {
    // TODO: Add assertions for input validation
  }

  RandomSample(const Tensor logits, Tensor out)
      : RandomSample{logits, out, std::nullopt, std::nullopt,
                     std::nullopt, std::nullopt} {}

  // Full interface with all parameters
  virtual void operator()(const Tensor logits, Tensor out,
                          std::optional<float> temperature,
                          std::optional<int> top_k,
                          std::optional<float> top_p,
                          std::optional<float> random_val) const = 0;

  // Simplified interface using constructor parameters
  virtual void operator()(const Tensor logits, Tensor out) const {
    return operator()(logits, out, temperature_, top_k_, top_p_, random_val_);
  }

 protected:
  float temperature_{1.0f};
  int top_k_{0};           // 0 = no limit
  float top_p_{1.0f};      // 1.0 = no limit
  float random_val_{0.0f}; // 0 = greedy (argmax)

  DataType logits_dtype_;
  DataType out_dtype_;
  Tensor::Size batch_size_{0};
  Tensor::Size vocab_size_{0};
};

}  // namespace infini::ops

#endif
```

---

## 需要确认的问题

### 问题 1: 输出类型

`out` 张量应该是 `int32` 还是 `int64`？

| 选项 | 优点 | 缺点 |
|------|------|------|
| A. `int32` | 内存占用小，与 FlashInfer 一致 | vocab_size 超过 2^31 时溢出 |
| B. `int64` | 兼容性更好，与 PyTorch argmax 一致 | 内存占用翻倍 |
| C. 模板参数 | 灵活，支持多种类型 | 增加代码复杂度 |

**InfiniCore 实现**: 支持多种整数类型 (I8, I16, I32, I64, U8, U16, U32, U64)
**vLLM 实现**: FlashInfer 返回 int32，但最终转为 int64

---

### 问题 2: 随机数生成

`random_val` 是单个种子值还是每个 batch 一个？

| 选项 | 说明 | 与现有实现一致性 |
|------|------|------------------|
| A. 单个 `float` | 整个 batch 使用同一个随机种子 | InfiniCore |
| B. 每请求独立 | 每个 batch 元素有独立的随机数 | vLLM |

```cpp
// 方案 A: 单个随机值
float random_val;  // 用于整个 batch

// 方案 B: 每请求独立随机数
Tensor random_vals;  // shape: [batch_size]
// 或
std::vector<float> random_vals;
```

---

### 问题 3: 是否返回 logprobs

vLLM 的 Sampler 会同时返回采样结果和 logprobs。

| 选项 | 说明 |
|------|------|
| A. 仅返回 indices | 与 InfiniCore 一致，简单 |
| B. 返回 indices + logprobs | 与 vLLM 一致，功能更完整 |

如果选择 B，可以考虑：
- 返回 `std::pair<Tensor, Tensor>` 或自定义结构体
- 或者单独创建 `RandomSampleWithLogprobs` 算子

---

### 问题 4: 批量参数支持

是否支持一个 batch 中不同请求使用不同的采样参数？

| 选项 | 说明 | 复杂度 |
|------|------|--------|
| A. 标量参数 | 整个 batch 使用相同参数 | 低 |
| B. 张量参数 | 每个 batch 元素有独立参数 | 高 |

```cpp
// 方案 A: 标量参数
float temperature;  // 所有请求共用

// 方案 B: 张量参数
Tensor temperatures;  // shape: [batch_size]
```

**vLLM 实现**: 支持混合采样 (同一 batch 中有 greedy 也有 random)

---

## 平台实现计划

| 平台 | 实现策略 | 优先级 |
|------|----------|--------|
| CPU | OpenMP 并行实现 | P0 |
| NVIDIA | CUDA kernel (参考 InfiniCore + CUB) | P0 |
| Iluvatar | 复用 CUDA 实现 | P1 |
| MetaX | 复用 CUDA 实现 | P1 |
| Moore | 复用 CUDA 实现 | P1 |
| Hygon | 复用 CUDA 实现 | P1 |
| Cambricon | CNNL + BANG kernel | P2 |
| Ascend | ACLNN API | P2 |

---

## 下一步

1. 确认上述 4 个问题的答案
2. 创建 `src/base/random_sample.h` 文件
3. 实现 CPU 版本 `src/cpu/random_sample/kernel.h`
4. 实现 NVIDIA 版本 `src/cuda/random_sample/kernel.h`
5. 添加单元测试

---

## 参考资料

- [InfiniCore random_sample 实现](../InfiniCore/src/infiniop/ops/random_sample/)
- [vLLM Sampler 文档](https://docs.vllm.ai/en/stable/api/vllm/v1/sample/sampler/)
- [CUB 库文档](https://nvlabs.github.io/cub/) (用于高效 GPU 排序)
