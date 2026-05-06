# Add 算子阅读路线

以 Add 算子的执行流程为线索，逐个文件理解 InfiniOps 的架构。

---

## 第 1 步：数据载体

### `src/tensor.h`

Tensor 是所有算子操作的基本单元。

关注点：
- Tensor 存了哪些信息（data、shape、stride、dtype、device）
- 构造函数有哪些重载
- `numel()`、`ndim()`、`IsContiguous()` 等方法
- `std::hash<Tensor>` 和 `std::equal_to<Tensor>` 特化（用于缓存 key）

### `src/data_type.h`

定义数据类型枚举（F16、F32、BF16 等）和类型映射。

关注点：
- `DataType` 枚举值
- `TypeMap`（编译期 C++ 类型 ↔ DataType 枚举的映射）
- `DataTypeMap`（为每种 DataType 存一个值）

### `src/device.h`

定义设备类型枚举和设备过滤。

关注点：
- `Device::Type` 枚举（kCpu、kNvidia、kCambricon 等）
- `EnabledDeviceFilter`（编译期决定哪些设备可用）
- `ActiveDevices` 类型列表

---

## 第 2 步：派发和调用机制

### `src/dispatcher.h`

核心派发机制，根据运行时值（dtype/device）跳转到编译期生成的代码。

关注点：
- `DispatchFunc` 模板：接收一个类型列表和运行时值，对每个类型生成一个 lambda 调用
- `Tag<T>` 结构：把类型包装成编译期常量
- 整个 switch-case 是怎么在编译期展开的

### `src/operator.h`

算子基类，提供 make/call/缓存的通用逻辑。

关注点：
- `OperatorBase`：持有 handle、stream、workspace
- `CacheKey`：用 shape/dtype/stride 做缓存 key
- `make()`：根据 Tensor 的 device 类型，构造对应平台的特化类
- `call()`：缓存查找 → 未命中则 make → 设置 handle/stream → 调用 operator()
- CRTP：`static_cast<const Key*>(this)` 实现编译期多态

### `src/base/add.h`

Add 算子的基类（你已经看过了）。

关注点：
- 构造函数缓存了哪些元信息
- `virtual void operator()(...) const = 0` 纯虚接口
- `protected` 成员变量的设计

---

## 第 3 步：平台实现（自顶向下）

### `src/nvidia/add/kernel.h`

NVIDIA 平台特化。

关注点：
- `NvidiaBackend` 结构体：定义了 malloc/memcpy/free/stream_t 等平台 API
- `template <> class Operator<Add, Device::Type::kNvidia>` 模板特化
- `using CudaAdd<add::NvidiaBackend>::CudaAdd` 继承构造函数

### `src/cuda/add/kernel.h`

CUDA 平台通用实现，与具体硬件无关。

关注点：
- `CudaAdd<Backend>` 模板类：构造时把 shape/stride 拷贝到 GPU
- `operator()` 实现：
  - `DispatchFunc<AllTypes>` 按数据类型派发
  - 根据数据类型 `reinterpret_cast` 指针
  - 计算 grid/block 维度
  - 根据最优 block_size 选择不同的 kernel 模板参数启动

### `src/cuda/add/kernel.cuh`

实际的 CUDA kernel 代码。

关注点：
- `AddKernel<T, BLOCK_SIZE>` 的实现
- 如何处理非连续 Tensor（通过 shape + stride 索引）
- CUDA 编程模型：thread、block、grid

---

## 调用链路总结

```
用户代码
  Operator<Add>::call(input, other, out)
    │
    ├─ CacheKey::Build(input, other, out)     // operator.h
    │   用 shape/dtype/stride 计算缓存 key
    │
    ├─ 缓存命中 → 直接取 op
    │  缓存未命中 → make(input, other, out)   // operator.h
    │     │
    │     └─ DispatchFunc<ActiveDevices>       // dispatcher.h
    │        根据 input.device().type() 派发
    │        │
    │        └─ new Operator<Add, kNvidia>     // nvidia/add/kernel.h
    │           继承 CudaAdd<NvidiaBackend>
    │           继承 Add (base/add.h)
    │
    ├─ op->set_handle/set_stream              // operator.h
    │
    └─ (*op)(input, other, out)               // CRTP 转发
        │
        └─ CudaAdd::operator()                // cuda/add/kernel.h
           │
           ├─ DispatchFunc<AllTypes>           // dispatcher.h
           │  根据 out.dtype() 派发
           │
           └─ AddKernel<T, BLOCK_SIZE><<<...>>>  // cuda/add/kernel.cuh
              GPU 上执行 elementwise 加法
```

---

## 每个文件的核心问题

看完每个文件后，试着回答：

1. **tensor.h**: Tensor 存了哪些信息？为什么 hash 不包含 data 指针？
2. **data_type.h**: TypeMap 是怎么在编译期把 C++ 类型映射到枚举值的？
3. **device.h**: EnabledDeviceFilter 是怎么根据编译宏决定启用哪些设备的？
4. **dispatcher.h**: DispatchFunc 是怎么用模板在编译期生成 switch-case 的？
5. **operator.h**: 缓存机制为什么要用 CacheKey 而不是直接用指针？
6. **nvidia/add/kernel.h**: Backend 模式的意义是什么？为什么不直接调用 CUDA API？
7. **cuda/add/kernel.h**: 为什么要根据 block_size 选择不同的 kernel 模板？
8. **cuda/add/kernel.cuh**: kernel 里是怎么处理非连续 Tensor 的？
