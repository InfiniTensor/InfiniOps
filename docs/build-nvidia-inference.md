# Reproducible NVIDIA inference SDK inputs

This recipe describes the Linux x86-64, CPython 3.12, CUDA 12.8, SM80,
`_GLIBCXX_USE_CXX11_ABI=0` combination used by the Torch-free inference SDK.
Use the source revisions pinned by the consuming InfiniCore checkout, not
floating branches. Other architectures/toolkits require separate validation.
The Python packages below supply compiled libraries and metadata; LM does not
execute their Python inference APIs. No `flashinfer-python` or Torch is required.

## Resolved third-party inputs

| Input | Exact artifact | SHA256 |
|---|---|---|
| Sampling | `flashinfer_jit_cache-0.6.7+cu128-cp39-abi3-manylinux_2_28_x86_64.whl` | `40139f1aab5435d9cb5ac0261883bf26740ba26680a9e2c9d7703168a6b4648d` |
| TVM FFI | `apache_tvm_ffi-0.1.10-cp312-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl` | `73d478395a8625dd92fde7b7fd92b4719f18f480b78336e422cb66cc7985213d` |
| FlashAttention | `flash_attn-2.8.3.tar.gz` | `1e71dd64a9e0280e0447b8a0c2541bad4bf6ac65bdeaa2f90e51a9e57de0370d` |
| CUTLASS | commit `087c84df83d254b5fb295a7a408f1a1d554085cf` archive | `99340f7a8c87f5295b91aecf8295df6693ad954807672c3869a7fb3d0084f709` |

Installed library hashes for the two wheels above:

```text
4aa059811da929fac4a4b25b26153649965d9d84ce35f29f06119fa874846c77  sampling.so
7657c74066e959a038324d618d92647a23c79433b3f7a9fbf4fd6354c4d3f9be  libtvm_ffi.so
```

The YAML version ranges describe the provider's accepted packages, not a lock
for a release. A release using a different wheel must record its resolved
version/hash and run the corresponding direct operator tests again.

## Acquire and install the fixed inputs

Run in a new working directory. Set `CORE` to an initialized recursive
InfiniCore checkout and use a new, absolute installation prefix:

```bash
set -euo pipefail
export CORE=/absolute/path/to/InfiniCore
export INFINI_ROOT="$PWD/sdk"
export CUDA_HOME=/usr/local/cuda
export INFINI_RT_ROOT="$INFINI_ROOT"
mkdir artifacts
cd artifacts
curl -fL -o 'flashinfer_jit_cache-0.6.7+cu128-cp39-abi3-manylinux_2_28_x86_64.whl' \
  'https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.7/flashinfer_jit_cache-0.6.7%2Bcu128-cp39-abi3-manylinux_2_28_x86_64.whl'
curl -fL -O 'https://files.pythonhosted.org/packages/50/6f/23740f06829030704e6f8f1f7093a06b7a68f904baa40053a5f594705bae/apache_tvm_ffi-0.1.10-cp312-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl'
curl -fL -O 'https://files.pythonhosted.org/packages/3b/b2/8d76c41ad7974ee264754709c22963447f7f8134613fd9ce80984ed0dab7/flash_attn-2.8.3.tar.gz'
curl -fL -o cutlass.tar.gz 'https://github.com/NVIDIA/cutlass/archive/087c84df83d254b5fb295a7a408f1a1d554085cf.tar.gz'
sha256sum -c <<'SHA256'
40139f1aab5435d9cb5ac0261883bf26740ba26680a9e2c9d7703168a6b4648d  flashinfer_jit_cache-0.6.7+cu128-cp39-abi3-manylinux_2_28_x86_64.whl
73d478395a8625dd92fde7b7fd92b4719f18f480b78336e422cb66cc7985213d  apache_tvm_ffi-0.1.10-cp312-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
1e71dd64a9e0280e0447b8a0c2541bad4bf6ac65bdeaa2f90e51a9e57de0370d  flash_attn-2.8.3.tar.gz
99340f7a8c87f5295b91aecf8295df6693ad954807672c3869a7fb3d0084f709  cutlass.tar.gz
SHA256
tar -xzf cutlass.tar.gz
export INFERENCE_ARTIFACTS="$PWD"
cd ..
python3.12 -m venv build-env
source build-env/bin/activate
python -m pip install cmake ninja libclang numpy packaging scikit-build-core pybind11 pyyaml
python -m pip install --no-index --no-deps --target "$INFINI_ROOT/python" \
  "$INFERENCE_ARTIFACTS/flashinfer_jit_cache-0.6.7+cu128-cp39-abi3-manylinux_2_28_x86_64.whl" \
  "$INFERENCE_ARTIFACTS/apache_tvm_ffi-0.1.10-cp312-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"
install -d "$INFINI_ROOT/lib"
install -m644 "$INFINI_ROOT/python/flashinfer_jit_cache/jit_cache/sampling/sampling.so" "$INFINI_ROOT/lib/"
install -m644 "$INFINI_ROOT/python/tvm_ffi/lib/libtvm_ffi.so" "$INFINI_ROOT/lib/"
export PYTHONPATH="$INFINI_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Retain both complete package directories and their `.dist-info` directories;
the Ops resolver uses `importlib.metadata`, not just the two copied libraries.
The selected FlashInfer wheel is large because it includes other precompiled
kernels; those are not added as Ops providers by this recipe.

## Build the SDK

```bash
cmake -S "$CORE/submodules/InfiniRT" -B build/rt \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INFINI_ROOT" \
  -DCMAKE_INSTALL_LIBDIR=lib -DWITH_CPU=ON -DWITH_NVIDIA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0 \
  -DCMAKE_CUDA_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0
cmake --build build/rt -j4
cmake --install build/rt
cmake -S "$CORE/submodules/InfiniOps" -B build/ops \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INFINI_ROOT" \
  -DCMAKE_INSTALL_LIBDIR=lib -DINFINI_RT_ROOT="$INFINI_ROOT" \
  -DWITH_CPU=ON -DWITH_NVIDIA=ON -DWITH_TORCH=OFF -DWITH_LINKED=ON \
  -DAUTO_DETECT_DEVICES=OFF -DAUTO_DETECT_BACKENDS=OFF \
  -DGENERATE_OPERATOR_CALL_INSTANTIATIONS=ON -DGENERATE_PYTHON_BINDINGS=OFF \
  -DBUILD_NVIDIA_FLASH_INFERENCE=ON -DBUILD_NVIDIA_W4_MARLIN=ON \
  -DFLASH_ATTN_SOURCE_ARCHIVE="$INFERENCE_ARTIFACTS/flash_attn-2.8.3.tar.gz" \
  -DFETCHCONTENT_SOURCE_DIR_CUTLASS="$INFERENCE_ARTIFACTS/cutlass-087c84df83d254b5fb295a7a408f1a1d554085cf" \
  -DINFINI_OPS_OPS="$CORE/submodules/InfiniOps/scripts/configs/infiniops_ops_nvidia_dense_sampling_torch_free.json" \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0 \
  -DCMAKE_CUDA_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0 \
  -DPython_EXECUTABLE="$(command -v python)"
cmake --build build/ops -j4
cmake --install build/ops
```

Marlin is enabled here because the LM profile explicitly contains W4 operators.
A dense-only SDK can set it `OFF` and use an allowlist without `w4_marlin_*`;
then its CUDA >=12.3/C++20 check and provider registration are both omitted.
Build CCL separately using the Core instructions and the same ABI/prefix before
building LM. Pin and record the installed NCCL version as part of that deployment.

## Third-party notices and validation

Keep the TVM wheel's `apache_tvm_ffi-0.1.10.dist-info/licenses/LICENSE` and
`NOTICE`. The selected FlashInfer wheel omits a license directory: obtain the
Apache-2.0 `LICENSE` from its matching upstream release, not from a floating
branch. For example:

```bash
install -d "$INFINI_ROOT/share/licenses/native-sampling"
curl -fL 'https://raw.githubusercontent.com/flashinfer-ai/flashinfer/v0.6.7/LICENSE' \
  -o "$INFINI_ROOT/share/licenses/native-sampling/FlashInfer-LICENSE.txt"
install -m644 "$INFINI_ROOT/python/apache_tvm_ffi-0.1.10.dist-info/licenses/LICENSE" \
  "$INFINI_ROOT/share/licenses/native-sampling/TVM-FFI-LICENSE.txt"
install -m644 "$INFINI_ROOT/python/apache_tvm_ffi-0.1.10.dist-info/licenses/NOTICE" \
  "$INFINI_ROOT/share/licenses/native-sampling/TVM-FFI-NOTICE.txt"
# When bundling these DSOs into an LM wheel, include the SDK notices too.
export LM=/absolute/path/to/InfiniLM
cp "$INFINI_ROOT/share/licenses/native-sampling/"*.txt "$LM/python/infinilm/licenses/"
```

Preserve the separate FlashAttention/CUTLASS and Marlin notices described in
`THIRD_PARTY.txt` when distributing those compiled providers. Do not treat this
dependency recipe as model qualification: validate the installed SDK with a tiny
sampling call and an attention call, then import the consuming `_infinilm`.
Record source revisions, wheel/library hashes, compiler/driver versions, ABI,
architecture, configure commands and actual library origins with the result.
