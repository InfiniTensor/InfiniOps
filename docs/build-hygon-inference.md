# Hygon inference SDK build

This recipe builds the InfiniRT/InfiniOps/InfiniCCL stack for Hygon DCUs. It
was validated on DTK 26.04 with `gfx936` (BW) devices; other Hygon
architectures need their own validation and their own `HYGON_ARCH`. It is a
build recipe, not a toolkit qualification report. Use the source revisions
pinned by the consuming InfiniCore checkout.

## Prerequisites

```sh
source /opt/dtk/env.sh
export CUDA_HOME=/opt/dtk/cuda/cuda
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
```

Consumers that bind Torch must match its C++ ABI (`_GLIBCXX_USE_CXX11_ABI`).
DTK's Triton compiles the HIP target of the device (for example `gfx936`,
wave size 64) on first use; CUDA compute-capability arithmetic does not apply.

## InfiniRT

```sh
cmake -S submodules/InfiniRT -B build/rt \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INFINI_ROOT" \
  -DWITH_CPU=ON -DWITH_HYGON=ON
cmake --build build/rt -j 12
cmake --install build/rt
```

## InfiniOps

`INFINI_OPS_OPS` selects what the build generates. The profile
`scripts/configs/infiniops_ops_hygon_infinilm.json` lists every operator the
InfiniLM model registry references: with the providers of this platform, or
as `public_api_only` where Hygon has none. The latter keep the complete
InfiniLM source compiling while those model families report unsupported at
runtime; they install no substitute kernels. Update the profile when InfiniLM
references another operator or Hygon gains a provider.

```sh
cmake -S submodules/InfiniOps -B build/ops \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INFINI_ROOT" \
  -DINFINI_RT_ROOT="$INFINI_ROOT" -DWITH_CPU=ON -DWITH_HYGON=ON \
  -DWITH_TRITON=ON -DHYGON_ARCH=gfx936 \
  -DGENERATE_PYTHON_BINDINGS=ON -DGENERATE_OPERATOR_CALL_INSTANTIATIONS=ON \
  -DAUTO_DETECT_DEVICES=OFF -DAUTO_DETECT_BACKENDS=OFF \
  -DINFINI_OPS_OPS="$PWD/scripts/configs/infiniops_ops_hygon_infinilm.json"
cmake --build build/ops -j 12
cmake --install build/ops
```

The Python `infini` package must remain importable: its JIT compiler and
kernel sources stay in use after installation. A CMake install puts the
package contents at the prefix; a wheel installs the namespace normally.

## InfiniCCL

```sh
cmake -S submodules/InfiniCCL -B build/ccl \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INFINI_ROOT" \
  -DWITH_HYGON=ON -DWITH_NCCL=ON \
  -DAUTO_DETECT_DEVICES=OFF -DBUILD_EXAMPLES=OFF
cmake --build build/ccl -j 12
cmake --install build/ccl
```

## Coverage

With this profile the operators of the dense transformer serving chain, the
standard MoE routing and the fused MoE — native `gfx936` kernels where the
architecture provides them, Triton JIT kernels otherwise — and the Clover
block-diffusion operators have Hygon providers. The model families whose
operators are `public_api_only` in the profile compile but report
unsupported; see the profile for the list. Model-level integration and
serving are documented in the InfiniLM repository.
