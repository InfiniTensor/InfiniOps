# InfiniOps

InfiniOps is a high-performance, hardware-agnostic operator library.

## 🛠️ Prerequisites

Ensure your environment meets the following requirements based on your target backend:

 - C++17 compatible compiler
 - CMake 3.18+
 - Hardware-specific SDKs (e.g., CUDA Toolkit)

---

## ⚙️ Installation & Building

InfiniOps uses CMake to manage backends.

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

| Option                                 | Functionality                      | Default
|----------------------------------------|------------------------------------|:-:
| `-DWITH_CPU=[ON\|OFF]`                 | Compile the CPU implementation     | n
| `-DWITH_NVIDIA=[ON\|OFF]`              | Compile the NVIDIA implementation  | n
| `-DWITH_ILUVATAR=[ON\|OFF]`            | Compile the Iluvatar implementation| n
| `-DWITH_HYGON=[ON\|OFF]`               | Compile the Hygon implementation   | n
| `-DWITH_METAX=[ON\|OFF]`               | Compile the MetaX implementation   | n
| `-DGENERATE_PYTHON_BINDINGS=[ON\|OFF]` | Generate Python bindings           | n

*Note: If no accelerator options are provided, `WITH_CPU` is enabled by default.*

For Hygon builds, set `DTK_ROOT` to the DTK installation root if it is not
installed at `/opt/dtk`. You can override the default DCU arch with
`-DHYGON_ARCH=<arch>` when configuring CMake.

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

Run the pybind11 example:

```bash
PYTHONPATH=src python ../examples/gemm.py
```
