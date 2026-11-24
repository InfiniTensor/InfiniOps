# Scalar multiplication and CPU cast regression checks

Source the personal SDK environment (`INFINI_ROOT`, `INFINILM_CXX11_ABI`), then:

```sh
cmake -S tests/native/runtime_contracts -B build/runtime-contracts -DCMAKE_BUILD_TYPE=Release
cmake --build build/runtime-contracts -j4
ctest --test-dir build/runtime-contracts --output-on-failure
```

Add `-DWITH_HYGON=ON -DCUDAToolkit_ROOT=/opt/dtk/cuda/cuda` for Hygon checks.
CPU-only builds do not require a GPU toolkit or GPU allocation. These tests
exercise the installed public Ops API, cached scalar calls, strided output
canaries, exact/partial storage alias rejection and rank-changing CPU Cast.
They are correctness regressions, not performance acceptance benchmarks.
