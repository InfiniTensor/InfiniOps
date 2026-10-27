# R5 kernel migration for Iluvatar

This change adapts four optimizations from the previously tested R5 profile
to the current public operator interfaces. It does not load legacy interposer
libraries or replace the model's execution runtime.

* `Add` skips device metadata allocation and host copies when all operands
  have the same contiguous shape. Broadcast and strided layouts keep the
  existing path and arithmetic.
* `SiluAndMul` assigns a contiguous BF16 row to blocks of 256 threads, with
  four columns per thread. Tail columns are checked. The existing FP32
  arithmetic is shared by both paths; overlapping buffers and other layouts
  retain the generic implementation.
* `Softmax` selects 128, 256, 512 or 1024 threads for contiguous BF16 rows on
  the last axis. It computes row offsets directly and avoids copying shape
  metadata. It retains current-main FP32 reductions and output conversion.
  Reduction ordering can change, so numerical and model regressions must be
  checked before interpreting a speedup.
* `RotaryEmbedding` retains R5's 64-thread block for each pair of Q/K heads
  in the qualified 28-head, 128-dimensional BF16 case. It reuses the public
  provider's arithmetic and combined cache, with checks for separate storage
  or disjoint slices of a packed QKV row. Other layouts retain the token-block
  implementation.

Current InfiniLM no longer calls `CausalSoftmaxInfinilm` for this path. Its
historical R5 row-bucket kernel is therefore not copied into this change.
The old benchmark numbers do not describe these migrated kernels.

## Capability query regression

The native Iluvatar `FlashAttnVarlenFunc` provider supports causal paged
inference only. It previously inherited the broader base-class `supports`
check, which accepted flat attention and later failed an execution assertion.
The provider now reports its supported subset before dispatch, allowing
InfiniLM's existing static attention fallback to run.

The metadata-only regression program requires an Iluvatar SDK build:

```sh
c++ -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -I"$INFINI_ROOT/include" \
  tests/iluvatar_varlen_support.cc -L"$INFINI_ROOT/lib" \
  -Wl,-rpath,"$INFINI_ROOT/lib" -linfiniops -linfinirt -o /tmp/varlen-support
/tmp/varlen-support
```

Use the C++ ABI selected when building the SDK. This checks rejection of flat,
noncausal and finite-window calls and acceptance of supported paged metadata.

## Operator regressions

The Python tests cover Softmax thread boundaries, in-place output, non-final
axes and strided rows, and SwiGLU widths immediately around a 1024-column tile.
Run them with the newly built Python package:

```sh
python -m pytest tests/test_add.py tests/test_softmax.py \
  tests/test_silu_and_mul.py tests/test_rotary_embedding.py --devices iluvatar -q
```

Measure complete model requests separately. Operator timing alone does not
establish end-to-end throughput or output quality.

## Standalone SDK qualification

On TG-V200 with COREX 4.3, the standalone public-API probe passed 149 cases.
All 149 output byte hashes matched a compatibility baseline built from
`1ca4777610d77e9063fffdb975ab35da8466104d` with the capability-query fix above.
The candidate was built from `640475a658de1f26bbf7c018fd0e0f8d5303a74c` plus
this patch. This is a bounded regression result, not universal bitwise
equivalence. Tests include FP32/FP16/BF16, in-place Softmax, causal-mask inputs,
strided fallbacks, SwiGLU tails, Add broadcasting, and packed/separate Q/K
RoPE with both rotation conventions and inverse rotation.

The probe uses Torch only for allocation, streams and reference calculations;
the tested operators are linked from the selected native SDK. This avoids
accidentally importing an older installed `infini.ops` Python extension.
Select one idle GPU with `CUDA_VISIBLE_DEVICES` and `IX_VISIBLE_DEVICES` so
that it is logical device zero, then run:

```sh
c++ -std=c++17 -shared -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0 \
  $(python3 -m pybind11 --includes) -I"$INFINI_ROOT/include" \
  -I"$CUDA_HOME/include" tests/standalone/r5_operator_probe.cc \
  -L"$INFINI_ROOT/lib" -Wl,-rpath,"$INFINI_ROOT/lib" \
  -linfiniops -linfinirt \
  -o "tests/standalone/r5_operator_probe$(python3-config --extension-suffix)"
python tests/standalone/check_iluvatar_r5.py --output operators.json
```

Use the ABI and Python headers of the selected SDK. Repeat with the baseline
SDK and compare the ordered `sha256` fields. The ordinary pytest cases added
to the operator test modules are intended for the repository's platform CI;
they do not substitute for running this SDK probe against the built library.

## Current-master performance comparison

`tests/standalone/benchmark_iluvatar_r5.cc` measures eight synthetic BF16
workloads for Softmax, SwiGLU and Add through the public operator API. It uses
the COREX runtime directly, without a Python/Torch allocation or stream setup.
The binary checks its loaded InfiniOps library against `INFINI_ROOT`, compares
results with a CPU float reference, and reports output checksums and five
timing samples of twenty calls after three warmups.

```sh
c++ -std=c++17 -O2 -D_GLIBCXX_USE_CXX11_ABI=0 \
  -I"$INFINI_ROOT/include" -I"$CUDA_HOME/include" \
  tests/standalone/benchmark_iluvatar_r5.cc \
  -L"$INFINI_ROOT/lib" -L"$CUDA_HOME/lib64" \
  -Wl,-rpath,"$INFINI_ROOT/lib" -linfiniops -linfinirt -lcudart \
  -o /tmp/benchmark-iluvatar-r5
LD_LIBRARY_PATH="$INFINI_ROOT/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH" \
  /tmp/benchmark-iluvatar-r5 > operator-timing.jsonl
```

Run each SDK in a separate process on the same idle device. Select one device
using `CUDA_VISIBLE_DEVICES` and `IX_VISIBLE_DEVICES` before execution. The
qualified build uses C++ ABI 0; match the actual SDK ABI when rebuilding.
The control is current master `640475a` plus only the capability-query fix,
commit `4038a8e`; the candidate adds the R5 kernels at `6489ac7`. Thus this
comparison holds the underlying Ops version constant.

Wall time includes operator dispatch and the final synchronization. Device
event intervals may include host submission gaps, so neither number is
described as an isolated kernel duration. The FNV-1a output checksum is a
regression aid for these synthetic inputs, not a cryptographic proof or a
claim of exactness for every shape. Numerical acceptance uses
`absolute_error <= 1e-6 + 0.01 * abs(reference)` for finite outputs.

Measured on one idle TG-V200 with COREX 4.3 on 2026-10-27. All eight cases
passed the numerical reference check and had matching output checksums.

| Operator | Shape | Control (µs/call) | R5 (µs/call) | Speedup |
| --- | --- | ---: | ---: | ---: |
| softmax | 16 × 128 | 16.299 | 11.381 | 1.43× |
| softmax | 512 × 512 | 40.807 | 20.918 | 1.95× |
| softmax | 2048 × 2048 | 121.963 | 84.039 | 1.45× |
| swiglu | 4 × 18944 | 18.898 | 14.561 | 1.30× |
| swiglu | 512 × 18944 | 206.588 | 80.091 | 2.58× |
| swiglu | 2048 × 18944 | 738.977 | 311.916 | 2.37× |
| add | 16 × 3584 | 18.649 | 16.447 | 1.13× |
| add | 2048 × 3584 | 99.830 | 97.064 | 1.03× |

Softmax and SwiGLU retain useful gains in these shapes. Add's measured gain is
smaller and does not establish a consistent large-shape improvement. These
are bounded synthetic operator results, not whole-model or universal gains.
Individual samples, SDK hashes and the benchmark source hash are recorded in
[`iluvatar-r5-benchmark.json`](iluvatar-r5-benchmark.json). Whole-model
measurements and output checks are in [InfiniLM PR #601](https://github.com/InfiniTensor/InfiniLM/pull/601).
