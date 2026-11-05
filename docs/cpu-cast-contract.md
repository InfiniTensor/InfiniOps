# CPU Cast storage contract

The native CPU `Cast` provider rejects overlapping input/output storage before
writing. An exact same view (pointer, dtype, shape and strides) is a no-op.
The check runs both at provider selection and at invocation: cached descriptors
reuse tensor geometry without including data pointers.

The provider retains the existing numerical conversion and dtype set.
Contiguous, transposed and gapped positive-stride views are supported, as are
read-only broadcast inputs. A writable output must have a non-overlapping
layout. Empty tensors need no storage; scalar tensors contain one element.
Shape/device mismatches, negative strides, unaligned/null nonempty storage and
overflowing extents are rejected. Address ranges are bounding spans, so some
interleaved disjoint views of the same allocation are conservatively rejected.
The layout test is also conservative for irregular interleaved output strides.
As elsewhere in the TensorView API, the caller owns the allocation and must
provide enough storage; the provider cannot establish allocation capacity from
a raw pointer.

## Regression

Before this fix, casting a two-element FP32 input into a same-address FP16
output overwrote unread source storage. This also caused InfiniLM's
`test_checkpoint_cast_rejects_bad_contract_before_writing` to fail. The original
behavior was reproduced before rebuilding the same installation with the fix.

CPU operator tests:

```sh
python -m pytest tests/test_cpu_cast.py tests/test_cast.py --devices cpu -q
```

On 2026-10-27, GCC 9.4, Python 3.10 and C++ ABI 0: **288 passed, 19 skipped**.
The skips require the optional Torch slot 8, which was not enabled in the
Cast-only CPU build. New tests explicitly select native slot 0. They cover
nine dtypes across three layouts, scalar/empty/no-op behavior, broadcast input,
overlap after a cached call, internal output overlap and shape rejection.

Build the selected operator and its Python bindings in a private prefix:

```sh
cmake -S . -B build-cpu-cast \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$CAST_PREFIX/infini" \
  -DINFINI_RT_ROOT="$INFINI_RT_ROOT" \
  -DAUTO_DETECT_DEVICES=OFF -DAUTO_DETECT_BACKENDS=OFF \
  -DWITH_CPU=ON -DWITH_TORCH=OFF -DINFINI_OPS_OPS=cast \
  -DGENERATE_PYTHON_BINDINGS=ON
cmake --build build-cpu-cast -j4
cmake --install build-cpu-cast
PYTHONPATH="$CAST_PREFIX:$PYTHONPATH" \
  python -m pytest tests/test_cpu_cast.py tests/test_cast.py --devices cpu -q
```

Match the C++ ABI and any SDK header/library paths to the installed InfiniRT.
The test environment used `-D_GLIBCXX_USE_CXX11_ABI=0`; its InfiniRT also included
Iluvatar, so COREX headers and runtime libraries were needed even for CPU tests.

The C++ contract executable additionally checks cached-descriptor metadata
changes, invalid addresses, negative/overflowing strides and null empty views:

```sh
c++ -std=c++17 -I"$CAST_PREFIX/infini/include" -I"$INFINI_RT_ROOT/include" \
  tests/check_cpu_cast_contract.cc -L"$CAST_PREFIX/infini/lib" \
  -L"$INFINI_RT_ROOT/lib" -Wl,-rpath,"$CAST_PREFIX/infini/lib" \
  -linfiniops -linfinirt -o /tmp/check-cpu-cast
/tmp/check-cpu-cast
```

This executable passed against the fixed SDK. An independent R5 SDK rebuild
with the same CPU header also passed all **12 InfiniLM checkpoint tests**.
The pre-existing R5 installation and original model files were not modified.
One TG-V200 model regression (batch 4, input 512, output 8; two warmups and
three requests) also matched the prior R5 output hash on every request. Of
1,359 runtime source files compared with that R5 build, only the CPU Cast
header changed. Counts and hashes are in `cpu-cast-validation.json`; this
regression does not replace or reattribute the earlier R5 performance results.

## Error handling

Unsupported requests fail with the existing operator factory's
`std::invalid_argument`/Python `ValueError` contract. Direct construction and
cached calls use the same runtime rejection as the existing NVIDIA Cast
provider. Debug-only assertions would allow Release builds to overwrite the
source, so rejection must remain active in Release builds. No conversion math,
GPU kernel or shared dispatch behavior is changed.
