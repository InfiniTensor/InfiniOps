# RMSNorm 设计文档

## 1. 算子接口

### 1.1 函数签名

```cpp
at::Tensor rms_norm(const at::Tensor &input, const at::Tensor &weight, double eps);
```

### 1.2 参数说明

| 参数名 | 类型 | 输入/输出 | 支持的数据类型 | 描述 | 约束条件 |
|--------|------|-----------|---------------|------|----------|
| input | at::Tensor | 输入 | float16/float32 | 输入 tensor，shape `[*, hidden_dim]` | 最后一维为归一化维度 |
| weight | at::Tensor | 输入 | float16/float32 | 权重 tensor，shape `[hidden_dim]` | 与 `input` 最后一维等长 |
| eps | double | 输入 | — | 数值稳定性常量 | 默认 1e-6 |
| output | at::Tensor | 输出 | float16/float32 | 输出 tensor，shape 同 `input` | dtype 同 `input` |

### 1.3 支持的数据类型

- [x] float16
- [x] float32

### 1.4 PyTorch 参考

```python
torch.nn.functional.rms_norm(input, normalized_shape, weight, eps)
```

InfiniOps 基类：`src/base/rms_norm.h`，成员 `dim_`（hidden_dim）、`batch_size_`、`nhead_`、`eps_`。

---

## 2. 计算逻辑

### 2.1 算法描述

RMSNorm 对输入 tensor 的每一行（最后一维）做 Root Mean Square 归一化：

$$y_i = x_i \cdot \text{rsqrt}\left(\frac{1}{N}\sum_{j=0}^{N-1} x_j^2 + \varepsilon\right) \cdot w_i$$

其中 $N$ = `hidden_dim`。

分步：
1. 对每行 $x$ 计算元素平方 $x^2$。
2. 沿行方向归约求和 $\text{sum} = \sum x^2$。
3. 计算均值 $\text{mean} = \text{sum} / N$。
4. 加 epsilon 并取 rsqrt：$\text{scale} = \text{rsqrt}(\text{mean} + \varepsilon)$。
5. 逐元素乘以 scale 和 weight：$y = x \cdot \text{scale} \cdot w$。

### 2.2 AscendC API 调用伪代码

```cpp
// 对每行 hidden_dim 个元素（x 已在 UB 中，float32）：

// Step 1: 计算 x²。
Mul(sqBuf, xBuf, xBuf, hiddenDim);

// Step 2: 归约求和。
// ReduceSum 结果存入 sumBuf（至少 32B）。
WholeReduceSum(sumBuf, sqBuf, hiddenDim, 1, 1, 8);

// Step 3-5: 标量运算（在 32B 对齐的 sumBuf 上操作）。
Muls(sumBuf, sumBuf, 1.0f / hiddenDim, 8);   // mean = sum / N
Adds(sumBuf, sumBuf, eps, 8);                  // mean + eps
Rsqrt(sumBuf, sumBuf, 8);                      // rsqrt(mean + eps)

// Step 6: 广播乘以 scale。
float scale = sumBuf.GetValue(0);
Muls(outBuf, xBuf, scale, hiddenDim);          // y = x * scale

// Step 7: 逐元素乘以 weight。
Mul(outBuf, outBuf, weightBuf, hiddenDim);      // y = y * weight
```

**FP16 输入时**，在 Step 1 之前插入升精度，在 Step 7 之后插入降精度：

```cpp
// 升精度：fp16 → fp32
Cast(xBufFp32, xBufFp16, RoundMode::CAST_NONE, hiddenDim);

// ... Steps 1-7 在 fp32 上执行 ...

// 降精度：fp32 → fp16
Cast(outBufFp16, outBufFp32, RoundMode::CAST_ROUND, hiddenDim);
```

### 2.3 实现路径选择

- [x] AscendC Kernel（纯 vector 实现）
- [ ] CATLASS 模板库（矩阵乘法类）
- [ ] ACLNN 封装（CANN 内置算子）

**选择理由**：RMSNorm 是纯 vector 归约 + 逐元素运算，不涉及矩阵乘法。CANN 的 `aclnnRmsNorm` 内部分解为 5 个子算子（Pows + ReduceMean + Add + Rsqrt + Mul），产生 inter-op 调度开销。自定义 AscendC kernel 可以将整个计算融合在单个 kernel 内，消除子算子之间的调度开销并实现 UB 内数据复用。

---

## 3. Tiling 策略

**算子类型**: Row-reduction（沿最后一维归约，输出与输入同形）

### 核心设计

RMSNorm 以**行**为处理单元。每行 `hidden_dim` 个元素必须整体装入 UB 才能完成归约。因此：

- **Block 级 Tiling**：将总行数分配到多核并行。
- **UB 级 Tiling**：每次处理一行（`tileLength = hiddenDim`）。核内循环遍历分配给该核的所有行。

```
GM:   [row 0] [row 1] ... [row M-1]     (M = totalRows)
        │       │             │
  ┌─────┘       │             └─────┐
  ▼             ▼                   ▼
Core 0       Core 1  ...        Core 39    ← Block 级（行分配）
 rows[0..k]  rows[k+1..2k]     rows[..]

Core 内：
  for each row:
    CopyIn(row)     ← GM → UB
    Compute(row)    ← reduction + scale + weight mul
    CopyOut(row)    ← UB → GM
```

### 3.1 Tiling 参数结构体

```cpp
struct RmsNormTilingData {
    int64_t totalRows;       // 总行数 = product(shape[:-1])
    int64_t hiddenDim;       // 最后一维长度 N
    int64_t hiddenDimAlign;  // 32B 对齐后的 N

    int64_t formerNum;       // 整核数量
    int64_t formerLength;    // 整核处理的行数
    int64_t tailNum;         // 尾核数量
    int64_t tailLength;      // 尾核处理的行数

    float eps;               // epsilon
    int64_t dtypeSize;       // 每个元素字节数（2 或 4）
};
```

### 3.2 Block 级 Tiling（核间切分）

按行数均匀分配到 `CORE_NUM` 个核，使用整核/尾核策略：

| 参数 | 计算公式 |
|------|----------|
| totalRows | product(input.shape[:-1]) |
| formerNum | totalRows % CORE_NUM（== 0 时取 CORE_NUM） |
| tailNum | CORE_NUM - formerNum |
| formerLength | totalRows / CORE_NUM + 1 |
| tailLength | totalRows / CORE_NUM |

**验证**：`formerNum * formerLength + tailNum * tailLength == totalRows`

### 3.3 UB 级 Tiling（核内切分）

每次处理一行。`tileLength = hiddenDim`（整行装入 UB）。

#### 精度处理

| 输入类型 | 计算精度 | UB 额外开销 |
|----------|----------|-------------|
| float32 | float32 | 无 |
| float16 | **升精度到 float32** | 需要 fp32 计算 buffer |

#### UB 分配表 — float32

| Buffer 名称 | 大小（字节） | 数量 | 用途 | 总大小 |
|-------------|-------------|------|------|--------|
| inQueueX | hiddenDim × 4 | 2 (double buf) | 输入行 | hiddenDim × 8 |
| outQueueY | hiddenDim × 4 | 2 (double buf) | 输出行 | hiddenDim × 8 |
| tmpBuf | hiddenDim × 4 | 1 | x² 中间结果 | hiddenDim × 4 |
| weightBuf | hiddenDim × 4 | 1 | weight（load once） | hiddenDim × 4 |
| sumBuf | 32 | 1 | 归约标量 | 32 |
| **总计** | | | | **hiddenDim × 24 + 32** |

**bufferCoefficient (fp32) = 24**

maxHiddenDim (fp32) = (UB_SIZE_LIMIT − 32) / 24

示例：UB = 192 KB → maxHiddenDim = 8191

#### UB 分配表 — float16

| Buffer 名称 | 大小（字节） | 数量 | 用途 | 总大小 |
|-------------|-------------|------|------|--------|
| inQueueX | hiddenDim × 2 | 2 (double buf) | 输入行 (fp16) | hiddenDim × 4 |
| outQueueY | hiddenDim × 2 | 2 (double buf) | 输出行 (fp16) | hiddenDim × 4 |
| xFp32Buf | hiddenDim × 4 | 1 | 升精度后的 x | hiddenDim × 4 |
| tmpFp32Buf | hiddenDim × 4 | 1 | x² 中间结果 | hiddenDim × 4 |
| weightFp32Buf | hiddenDim × 4 | 1 | weight (fp32, load once) | hiddenDim × 4 |
| sumBuf | 32 | 1 | 归约标量 | 32 |
| **总计** | | | | **hiddenDim × 20 + 32** |

**bufferCoefficient (fp16) = 20**

maxHiddenDim (fp16) = (UB_SIZE_LIMIT − 32) / 20

示例：UB = 192 KB → maxHiddenDim = 9828

#### 典型模型 hidden_dim 验证

| 模型 | hidden_dim | fp32 UB 使用 | fp16 UB 使用 | 是否 fit |
|------|-----------|-------------|-------------|---------|
| Qwen-7B | 4096 | 98,336 B (50%) | 81,952 B (42%) | ✓ |
| Llama-8B | 4096 | 98,336 B | 81,952 B | ✓ |
| Llama-70B | 8192 | 196,640 B (100.02%) | 163,872 B (83%) | fp16 ✓, fp32 需降为 BUFFER_NUM=1 |

**注意**：fp32 + hidden_dim=8192 超出 192KB 32 字节。此时 Host 端应检测并降低 BUFFER_NUM 为 1（bufferCoefficient 变为 16，maxHiddenDim = 12287）。

#### UB 约束验证

- **UB 对齐**：32 字节
- **hiddenDimAlign**：`((hiddenDim + alignElements − 1) / alignElements) * alignElements`，其中 `alignElements = 32 / dtypeSize`
- **UB 总使用** ≤ UB_SIZE_LIMIT（通过 `AscendC::GetSysWorkSpaceSize()` 运行时获取）

---

## 4. Workspace 需求

### 4.1 Workspace 大小

```cpp
size_t workspaceSize = sizeof(RmsNormTilingData);
```

Tiling 参数通过 workspace 传递给 kernel。

---

## 5. 性能优化

### 5.1 关键优化点

1. **单 kernel 融合**：将 CANN 的 5 个子算子（Pows + ReduceMean + Add + Rsqrt + Mul）融合为 1 个 kernel，消除 inter-op 调度开销。
2. **UB 数据复用**：输入行在 UB 中被读取一次，用于平方和归约，又用于 scale 乘法——无需重复从 GM 加载。
3. **Weight 一次加载**：weight 向量在 Init 阶段加载到 UB，后续所有行复用。
4. **Double buffer**：输入/输出使用 BUFFER_NUM=2，隐藏 GM 访存延迟。

### 5.2 算子特性

- **计算模式**: memory-bound（归约 + 逐元素乘法，计算强度低）
- **访存模式**: 顺序行访问（最后一维连续）
- **并行性**: 高（行间完全独立）

---

## 6. Kernel 端实现要点

### 6.1 Init（核内初始化）

```cpp
__aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR y,
                            GM_ADDR workspace, GM_ADDR tiling) {
    // 1. 从 tiling workspace 读取 RmsNormTilingData。
    // 2. 判断当前 block 是整核还是尾核，计算行偏移和行数。
    // 3. 设置 xGm / yGm 的 GlobalBuffer。
    // 4. 加载 weight 到 weightBuf（仅一次）。
    //    - fp16 输入时：加载 weight_fp16 → cast 到 weightFp32Buf。
    //    - fp32 输入时：直接加载到 weightBuf。
    // 5. 初始化 pipe / queue。
}
```

### 6.2 执行流程（核内循环）

```cpp
__aicore__ inline void Process() {
    // coreRows = 当前核分配的行数
    for (int64_t row = 0; row < coreRows; ++row) {
        CopyIn(row);
        Compute(row);
        CopyOut(row);
    }
}
```

### 6.3 CopyIn

```cpp
__aicore__ inline void CopyIn(int64_t row) {
    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    DataCopy(xLocal, xGm[row * hiddenDim], hiddenDim);
    inQueueX.EnQue(xLocal);
}
```

### 6.4 Compute

```cpp
__aicore__ inline void Compute(int64_t row) {
    LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();

    // [fp16 only] Cast x to fp32.
    // Cast(xFp32, xLocal, CAST_NONE, hiddenDim);

    // Step 1: x².
    Mul(tmpBuf, xFp32, xFp32, hiddenDim);

    // Step 2: ReduceSum → sumBuf.
    // 使用 WholeReduceSum 或手动分块归约。

    // Step 3-5: mean → +eps → rsqrt（在 sumBuf 上操作）。
    Muls(sumBuf, sumBuf, 1.0f / hiddenDim, 8);
    Adds(sumBuf, sumBuf, eps, 8);
    Rsqrt(sumBuf, sumBuf, 8);
    float scale = sumBuf.GetValue(0);

    // Step 6: y = x * scale.
    Muls(yFp32, xFp32, scale, hiddenDim);

    // Step 7: y = y * weight.
    Mul(yFp32, yFp32, weightBuf, hiddenDim);

    // [fp16 only] Cast back to fp16.
    // Cast(yLocal, yFp32, CAST_ROUND, hiddenDim);

    inQueueX.FreeTensor(xLocal);
    outQueueY.EnQue(yLocal);
}
```

### 6.5 CopyOut

```cpp
__aicore__ inline void CopyOut(int64_t row) {
    LocalTensor<T> yLocal = outQueueY.DeQue<T>();
    DataCopy(yGm[row * hiddenDim], yLocal, hiddenDim);
    outQueueY.FreeTensor(yLocal);
}
```

---

## 7. 实现检查清单

### 7.1 文件结构

- [ ] `csrc/ops/rms_norm/CMakeLists.txt`
- [ ] `csrc/ops/rms_norm/op_host/rms_norm.cpp`
- [ ] `csrc/ops/rms_norm/op_kernel/rms_norm.cpp`
- [ ] `csrc/ops.h`（添加声明）
- [ ] `csrc/register.cpp`（添加 `m.def` + `m.impl`）
- [ ] `csrc/CMakeLists.txt`（添加 host + kernel 源文件）

### 7.2 Host 端实现

- [ ] 定义 `RmsNormTilingData` 结构体
- [ ] 计算 totalRows = product(input.shape[:-1])
- [ ] Block 级 Tiling 参数（formerNum/tailNum/formerLength/tailLength）
- [ ] 检测 UB 是否能容纳 hiddenDim（超限时降低 BUFFER_NUM）
- [ ] 分配 workspace 并拷贝 tiling data
- [ ] 调用 `EXEC_KERNEL_CMD(rms_norm, ...)`

### 7.3 Kernel 端实现

- [ ] Init：整核/尾核偏移计算，weight 加载
- [ ] CopyIn：GM → UB 行拷贝
- [ ] Compute：fp16 升精度 → x² → ReduceSum → rsqrt → scale → weight mul → fp16 降精度
- [ ] CopyOut：UB → GM 行写回
- [ ] Process：行循环

### 7.4 测试验证

- [ ] 小规模：shape `[4, 128]`，fp32/fp16
- [ ] 中等规模：shape `[32, 4096]`，fp32/fp16
- [ ] 大规模：shape `[128, 8192]`，fp16
- [ ] 正确性：与 `torch.nn.functional.rms_norm` 对比
- [ ] 边界：shape `[1, 128]`（单行）、`[1024, 128]`（多行少列）

---

## 8. 参考实现

- **InfiniOps 基类**: `src/base/rms_norm.h`
- **InfiniOps CANN 实现**: `src/ascend/rms_norm/kernel.h`（使用 `aclnnRmsNorm`）
- **PyTorch**: `torch.nn.functional.rms_norm`
- **有效输入范围**: 无限制（任意实数），eps > 0
