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
