# InfiniOps 性能评估脚本

独立于 pytest 的系统化性能评估工具，测量 InfiniOps 算子与 PyTorch 参考实现的耗时对比。

## 快速开始

```bash
# 列出可用算子和设备
python scripts/benchmark.py --list

# 快速验证单个算子
python scripts/benchmark.py --ops add --device cpu --mode quick --no-json

# GPU 上跑全部 ntops 算子（推荐）
python scripts/benchmark.py --category ntops --device cuda --mode standard --output ntops_results.json
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--ops` | 指定算子名（可多个） | 全部可用算子 |
| `--category` | `native` / `torch` / `ntops` / `all` | `all` |
| `--device` | 设备（可多个，如 `cuda cpu`） | 全部可用设备 |
| `--dtype` | 数据类型：`float16` `bfloat16` `float32` | fp16 + bf16 |
| `--mode` | `quick` / `standard` / `thorough` | `standard` |
| `--warmup` | Warmup 轮数 | `3` |
| `--min-time` | 最小测量时间（秒） | `0.1` |
| `--output` | JSON 输出路径 | 无（不输出 JSON） |
| `--json-only` | 仅输出 JSON，不打印表格 | 关闭 |
| `--no-json` | 仅打印表格，不输出 JSON | 关闭 |
| `--list` | 列出可用算子并退出 | 关闭 |

## 三种模式

| 模式 | Shape 数量 | Dtype | 用途 |
|------|-----------|-------|------|
| `quick` | 每算子 1 个 | fp16, bf16 | 快速验证流程 |
| `standard` | 2-3 个（LLaMA decode/prefill） | fp16, bf16 | 日常性能评估（推荐） |
| `thorough` | 4-7 个（全量 LLM shape） | fp16, bf16, fp32 | 完整性能报告 |

Shape 基于 LLaMA-7B / 70B 模型的真实推理维度（hidden=4096/8192, FFN=11008/28672 等）。默认 shape 足够大（64M 元素），确保 kernel 计算时间远大于 pybind11 调度开销。

## 算子类别

### Native（手写原生算子，14 个）

有专用 C++/CUDA 后端实现的算子：

`add` `mul` `cast` `cat` `gemm` `matmul` `linear` `rms_norm` `causal_softmax` `swiglu` `flash_attention` `rotary_embedding` `add_rms_norm` `reshape_and_cache`

### ntops（ATen Fallback 算子，50 个）

通过 ATen `_out` API（slot=8）分发的 LLM 推理常用算子，与 PyTorch 原生算子性能对比：

**逐元素一元**：`abs` `neg` `exp` `rsqrt` `sigmoid` `silu` `gelu` `tanh` `sin` `cos` `bitwise_not` `softmax` `sqrt` `reciprocal` `log_softmax` `ceil` `floor` `log` `sign` `round` `hardtanh`

**逐元素二元**：`add` `sub` `div` `pow` `eq` `ne` `lt` `le` `gt` `ge` `bitwise_and` `bitwise_or` `maximum` `minimum`

**归约**：`sum` `mean` `cumsum` `amax` `amin` `argmax`

**矩阵运算**：`mm` `bmm` `addmm`

**排序/索引**：`topk` `sort` `gather` `index_select`

**归一化**：`rms_norm`

**池化**：`avg_pool2d`

### Torch（全部 PyTorch Fallback 算子，500+ 个）

基于 `torch_ops_metadata.json` 的完整 ATen fallback 算子集。

## 性能对比方法

### 测量方式

使用 `torch.utils.benchmark.Timer.blocked_autorange()` 进行多次测量取中位数。InfiniOps 和 PyTorch ref 均使用 `out=` 预分配输出 API，避免 tensor 分配开销影响对比。

### 调度路径差异

- **InfiniOps**：Python → pybind11（~15-20us）→ C++ dispatch → ATen kernel
- **PyTorch**：Python → 原生 C binding（~5us）→ ATen kernel

两者底层调用相同的 ATen CUDA kernel，差异仅在前端调度开销。当 kernel 计算时间远大于 ~20us 时，speedup 趋近 1.00x。

### 已知特殊情况

| 算子 | 现象 | 原因 |
|------|------|------|
| `silu` `hardtanh` | speedup ~1.9x | PyTorch Python API 不支持 `out=`，ref 多了 tensor 分配+copy 开销 |
| `add` | speedup <1.0x | 走 native slot 0 自定义后端，op cache + descriptor 开销比 ATen slot 8 重 |
| `rms_norm` | speedup <1.0x | 走 native slot 0 自定义后端，原生 kernel 在大 tensor 下性能不如 PyTorch |

## 使用示例

```bash
# ntops 算子性能对比（推荐用法）
python scripts/benchmark.py --category ntops --device cuda --mode standard --output ntops_results.json

# 单个算子快速验证
python scripts/benchmark.py --ops sigmoid mm --device cuda --dtype float16 --mode quick --no-json

# 仅原生算子
python scripts/benchmark.py --category native --device cuda --mode standard --output native.json

# 全量测试（耗时较长）
python scripts/benchmark.py --mode thorough --device cuda --output full.json

# 调整测量参数
python scripts/benchmark.py --category ntops --warmup 5 --min-time 0.3 --output results.json
```

## 输出格式

### 控制台表格

```
==========================================================================================================
InfiniOps Performance Benchmark | Device: cuda | Mode: quick
==========================================================================================================
Category Operator             Shape                          dtype    InfiniOps(us)  PyTorch(us)   Speedup
----------------------------------------------------------------------------------------------------------
ntops    abs                  [8192,8192]                    float16         158.75       158.24     1.00x
ntops    silu                 [8192,8192]                    float16         174.29       329.52     1.89x
ntops    mm                   [128,4096]x[4096,11008]        float16         104.60       104.41     1.00x
ntops    rms_norm             [8192,8192]                    float16           1.05         2.33     0.45x
----------------------------------------------------------------------------------------------------------
Summary: 50 benchmarks (0 native + 0 torch ops) | Avg speedup: 1.01x
==========================================================================================================
```

### JSON 文件

```json
{
  "metadata": {
    "timestamp": "2026-05-26T...",
    "device": ["cuda"],
    "mode": "standard",
    "torch_version": "2.6.0+cu124"
  },
  "results": [
    {
      "category": "ntops",
      "operator": "abs",
      "device": "cuda",
      "dtype": "torch.float16",
      "shape_description": "[8192,8192]",
      "infiniops_median_us": 158.75,
      "reference_median_us": 158.24,
      "speedup": 1.00,
      "status": "ok"
    }
  ],
  "summary": {
    "total": 50,
    "ok": 50,
    "avg_speedup": 1.01
  }
}
```

## 错误处理

- **算子不可用**：自动跳过，标记 `SKIP`
- **dtype 不支持**：捕获 RuntimeError，跳过
- **GPU OOM**：捕获异常，跳过该配置
- **Metadata 不存在**：仅 benchmark 原生算子

## 文件说明

| 文件 | 说明 |
|------|------|
| `scripts/benchmark.py` | 主脚本（CLI、测量引擎、算子 setup、输出格式化） |
| `scripts/benchmark_configs.py` | 原生算子 LLM shape 配置、FLOPS 计算、标量默认值 |

## 依赖

- `torch`（含 `torch.utils.benchmark`）
- `infiniops`
- `tests/utils.py`（复用 `randn_strided`、`empty_strided`、`get_stream` 等工具函数）

## 前置条件

ntops 算子需要 `feat/torch-operator-bases` 分支并启用 `WITH_TORCH` 编译：

```bash
git checkout feat/torch-operator-bases
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) pip install . --no-build-isolation \
    -C cmake.define.WITH_CPU=ON \
    -C cmake.define.WITH_NVIDIA=ON
```
