# 知识库检索与精排的算子接入

目标是让 BGE embedding 和 BGE reranker 的数据计算可以复用 InfiniOps，在 CPU、NVIDIA、MLU 和昇腾上保持相同的公开接口。它不是智能体框架，也不实现文档管理、模型调度或 Graph runtime。

## 算子范围

- `OnnxGather` 使用 ONNX Gather-13 的轴插入语义，支持标量／多维／负索引。旧 `Gather` 保留 PyTorch 等秩索引语义，不能互换。
- `Cast` 的 ATen slot 8 覆盖整数／bool mask 转换与非连续视图；已有原生实现编号不变。
- `NativeLayerNorm` 复用既有 ATen 生成器，保留 output／mean／rstd 三个输出，不复制厂商实现。
- Matmul、Softmax、Add/Sub/Mul/Div、Erf、Tanh、Addmm 和布局算子沿用已有实现。导出图的 GELU 由 Div／Erf／Add／Mul 表达。分类头可由 Matmul＋Add 或 Addmm 表达；旧 `Gemm` 的 C 输入限制不在本次扩大。
- RMSNorm 修复公开工厂的输入约束。它不在上述两个固定 encoder 图内，但通用调用者不应依赖上层兜底才能避免越界。

`scripts/configs/infiniops_ops_retrieval_torch.json` 提供跨这四类设备的 ATen 检索构建清单。Cast 固定选择 slot 8，避免清单在没有原生 slot 0 的设备上配置失败；其他原生实现仍可由调用方按能力显式选择。检索清单选择既有 ATen Add slot 1，以避免原生广播 Add 在大形状上的逐元素慢通路；该实现现在和其他 ATen 算子一样使用调用方的 runtime stream。清单不改变默认构建及默认 dispatch 策略。

## 外部内存与流

InfiniRT 或上层程序负责内存所有权。ATen wrapper 只借用指针，保留 dtype／shape／stride／device，不从外部指针猜测设备，也不接管释放。注册后端首次使用时以零元素 factory 初始化一次，兼容旧 Torch；后续调用复用初始化状态。

昇腾的 storage descriptor 构建集中在 `torch/ascend/external_tensor.h`，标记为不可扩容，避免 `_out` 调用悄悄替换外部存储。公共转换只通过一个按设备类型定制的包装入口调用，不包含 NPU 数学计算分支。输出形状与容量仍由调用方正确提供；这里不从 C++ 函数签名推导数学形状。

ATen provider 在传入的 runtime stream 上提交。昇腾调用前根据 ACL 的实际设备对齐框架缓存，再由 StreamGuard 切换并恢复调用方设备／流。原生 runtime 不能自动感知 torch_npu 的异步 host task queue，因此启动进程前须设置：

```bash
export TASK_QUEUE_ENABLE=0
export INFINIOPS_PROVIDER_MODULES=torch_npu
```

检查使用 torch_npu SDK 导出的 `OpApiGetTaskQueueEnable()` 获取实际状态，不仅读取环境变量。初始化以后修改环境变量不能伪装队列已关闭；默认 stream 也不绕过此约束。对应 CI 配置已同步。

## 验证

```bash
# 按硬件选择 --devices：cpu cuda / mlu / npu。
python -m pytest tests/test_onnx_gather.py tests/test_cast.py tests/test_native_layer_norm.py -k 'onnx_gather or encoder_' --devices cpu cuda -q -rs
# 必需集成测试显式检查所有算子实现，不能以缺算子或设备不可用作为 skip 放行。
INFINI_OPS_INSTALL_PREFIX=<ops-prefix> INFINI_RT_ROOT=<rt-prefix> \
python -m pytest tests/test_retrieval_encoder.py tests/test_encoder_external_memory.py --devices cpu cuda -q -rs
# 昇腾专用：跨卡后恢复原设备／流，以及已缓存的 host queue 不能靠改环境变量绕过。
python -m pytest tests/test_ascend_external_stream.py --devices npu -q -rs
```

MLU／昇腾使用设备专用 SDK 时，应排除 `test_onnx_gather_rejects_out_of_range_indices`；该数据越界拒绝用例只在 CPU 执行，避免设备断言破坏后续测试进程。测试选择必须写明，不把未执行项记成通过。

检索组合链测试覆盖查表、LayerNorm、attention、mask、Erf GELU、CLS 与分类 logits，并与 PyTorch 参考计算比较。外部内存测试的所有输入、输出由 InfiniRT Malloc 分配，在原生 stream 上执行并复制回主机；另在冷进程验证首次使用，检查重复调用、可用双卡和过小输出拒绝。这些是算子级集成测试，不等于完整模型精度、性能或长稳验收。
