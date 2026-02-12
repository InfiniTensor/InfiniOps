# InfiniOps

InfiniOps is a high-performance, hardware-agnostic operator library. It provides a unified C-style interface for executing various operators.

Currently supported backends/accelerators include:

- CPU；
- CUDA
  - NVIDIA GPU；
  - MooreThreads GPU；
  - Iluvatar CoreX GPU；
  - MetaX GPU；
  - HYGON DCU；
- Ascend NPU；
- Cambricon MLU；
- Kunlun XPU；

API definitions and usage examples can be found in [`InfiniCore-Documentation`](https://github.com/InfiniTensor/InfiniCore-Documentation)。

## 🛠️ Prerequisites

Ensure your environment meets the following requirements based on your target backend:

 - C++17 compatible compiler
 - CMake 3.18+
 - Hardware-specific SDKs (e.g., CUDA Toolkit)

---

## ⚙️ Installation & Building

InfiniOps uses CMake to manage backends. Only one hardware accelerator backend can be enabled/compiled at a time.

### 1. Setup Environment

Ensure you have the corresponding SDK installed and environment variables set up for the platform/accelerator you are working on. 


### 2. Configure and Build
Using these commands at the root directory of this project: 

```bash
mkdir build && cd build

cmake .. <OPTIONS>

make -j$(nproc)
```
For the `<OPTIONS>`:

| Option                      | Functionality                     | 默认值
|-----------------------------|-----------------------------------|:-:
| `-DUSE_CPU=[ON\|OFF]`       | Compile the CPU implementation    | n*
| `-DUSE_NVIDIA=[ON\|OFF]`    | Compile the NVIDIA implementation | n
| `-DUSE_METAX=[ON\|OFF]`     | Compile the MetaX implementation  | n

*Note: If no accelerator options are provided, `USE_CPU` is enabled by default.


## 🚀 Running Examples
After a successful build, the executables are located in the `build/examples` directory.

Run the GEMM example:

```bash
./examples/gemm
```

Run the data_type example: 

```bash
./examples/data_type
```

Run the tensor example: 

```bash
./examples/tensor
```
