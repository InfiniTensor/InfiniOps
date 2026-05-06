# FlashInfer 在 vLLM 中的采样集成 — 阅读指南

按调用链从上到下，依次阅读以下文件即可理解完整的 FlashInfer 采样集成链路。

---

## 1. 顶层调度 — `vllm/vllm/v1/sample/sampler.py`

vLLM 的采样总入口，`Sampler` 类的 `forward()` 方法协调整个采样流程：

- 计算 logprobs（如需要）
- 应用 temperature 缩放
- 调用 `TopKTopPSampler` 做 Top-K/Top-P 过滤
- 处理 greedy vs random sampling 分支
- 将 token ID 转为 int32（FlashInfer 兼容性要求）

---

## 2. FlashInfer 集成层 — `vllm/vllm/v1/sample/ops/topk_topp_sampler.py`（最核心）

**这是最关键的文件**，包含 `TopKTopPSampler` 类和 `flashinfer_sample()` 函数。

核心逻辑：

- 通过 `VLLM_USE_FLASHINFER_SAMPLER=1` 环境变量控制是否启用 FlashInfer
- 检查 GPU compute capability 是否兼容，不兼容则 fallback 到原生实现
- `flashinfer_sample()` 中的分发逻辑：

```python
if k is None:      # top-p only
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    flashinfer.sampling.top_p_sampling_from_probs(probs, p, deterministic=True)

elif p is None:    # top-k only
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    flashinfer.sampling.top_k_sampling_from_probs(probs, k, deterministic=True)

else:              # top-k + top-p 组合
    flashinfer.sampling.top_k_top_p_sampling_from_logits(logits, k, p, deterministic=True)
```

---

## 3. FlashInfer Python API — `flashinfer/flashinfer/sampling.py`

FlashInfer 的 Python 层采样接口，关键函数：

- `top_p_sampling_from_probs()` — 从概率分布做 top-p 采样
- `top_k_sampling_from_probs()` — 从概率分布做 top-k 采样
- `top_k_top_p_sampling_from_logits()` — 从 logits 直接做 top-k + top-p（内含 softmax）
- `min_p_sampling_from_probs()` — 从概率分布做 min-p 采样
- `sampling_from_probs()` — 无过滤的纯采样

展示了 Python 参数如何传递到 C++ 层：seed/offset、deterministic 标志、per-batch 参数（tensor vs scalar）。

---

## 4. FlashInfer C++ 内核 — `flashinfer/include/flashinfer/sampling.cuh`

头文件模板函数，包含所有采样 kernel 的 host 函数签名：

- `SamplingFromProb<T, IdType>(probs, output, valid, indices, batch_size, d, deterministic, seed_arr, seed_val, offset_arr, offset_val, stream)`
- `TopKSamplingFromProb<T, IdType>(probs, output, valid, indices, top_k_arr, batch_size, top_k_val, d, ...)`
- `TopPSamplingFromProb<T, IdType>(probs, output, valid, indices, top_p_arr, batch_size, top_p_val, d, ...)`
- `MinPSamplingFromProb<T, IdType>(probs, min_p_arr, output, valid, indices, batch_size, min_p_val, d, ...)`
- `TopKTopPSamplingFromProb<T, IdType>(probs, top_k_arr, top_p_arr, output, valid, indices, batch_size, top_k_val, top_p_val, d, ...)`
- `OnlineSoftmax<DType>(logits, output, batch_size, d, temperature_arr, temperature_val, workspace, workspace_size, enable_pdl, stream)`

模板参数说明：
- `T` — 概率/数据类型（float、half、__nv_bfloat16）
- `IdType` — 索引类型（int32_t、int64_t）
- `DType` — OnlineSoftmax 的数据类型

---

## 5. 测试 — `vllm/tests/v1/sample/test_topk_topp_sampler.py`

对比原生 PyTorch 实现与 FlashInfer 实现的正确性测试，可用于验证集成是否正确。

---

## 推荐阅读顺序

1. **`topk_topp_sampler.py`** 的 `flashinfer_sample()` 函数（~50 行）— 理解分发逻辑
2. **`flashinfer/sampling.py`** — 理解 Python 层如何包装参数并调用 C++
3. **`sampler.py`** — 理解上层如何组织整个采样流程
4. **`sampling.cuh`** — 按需查阅 C++ 函数签名细节
