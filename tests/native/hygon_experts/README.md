# Hygon W8A8 experts

These checks call the installed SDK public `DeepseekV4W8a8Experts` interface.
They compare all output and workspace bytes with the retained scalar kernels,
check quantization against a separate CPU oracle and protect workspace guards.
Coverage includes routed/shared experts, FP32/BF16, changing graph inputs, ties,
invalid/duplicate expert IDs, short/tail dimensions, misaligned weight fallback,
the 16/17-token epilogue boundary, grid-stride loops and the safe INT32 dot limit.

Source the existing SDK environment, including `INFINI_ROOT` and
`INFINILM_CXX11_ABI`, and use the matching Hygon toolchain:

```sh
cmake -S tests/native/hygon_experts -B build/expert-checks \
  -DCMAKE_CUDA_COMPILER=/opt/dtk/cuda/cuda/bin/nvcc \
  -DCUDAToolkit_ROOT=/opt/dtk/cuda/cuda \
  -DCMAKE_CUDA_ARCHITECTURES=OFF \
  -DCMAKE_CUDA_FLAGS="-arch=gfx936" -DCMAKE_BUILD_TYPE=Release
cmake --build build/expert-checks -j4
python tests/native/hygon_experts/check_experts.py \
  --library build/expert-checks/libexpert_probe.so --output verification-experts
```

Add `--performance` for five alternating timing rounds against the retained
repository scalar implementation. This local baseline is explicitly distinct
from SGLang and the owner's indicator-10 acceptance suite. No acceptance claim
is made by a successful microbenchmark.
