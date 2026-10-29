# Hygon Clover operators

These operators support the companion InfiniLM Clover implementation. Model
forward stays in C++; Hygon driver modules launch warmed Triton binaries.
Compilation uses the installed DTK Triton HIP target (for example `gfx936`,
wave size 64), not CUDA compute-capability arithmetic.

## Build

Install an InfiniRT revision containing the Hygon driver interface first
([InfiniRT PR #52](https://github.com/InfiniTensor/InfiniRT/pull/52)). Then:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INFINI_ROOT" -DINFINI_RT_ROOT="$INFINI_ROOT" \
  -DWITH_CPU=ON -DWITH_HYGON=ON -DHYGON_ARCH=gfx936 -DWITH_TRITON=ON \
  -DGENERATE_PYTHON_BINDINGS=ON -DAUTO_DETECT_DEVICES=OFF \
  -DAUTO_DETECT_BACKENDS=OFF
cmake --build build -j 12
cmake --install build
```

The Python `infini.triton` package must be importable. CMake installs its package
contents at the prefix, while wheel installation supplies the `infini` namespace.
Include DTK's CUDA compatibility library directory in `LD_LIBRARY_PATH`.

`INFINI_OPS_DECLARATIONS` accepts a semicolon-separated list of additional public
operator APIs. These can have no provider in the selected build: capability
queries return unsupported, and execution is rejected. This allows a downstream
application to compile its complete model registry without claiming that every
registered model runs on Hygon. InfiniLM supplies a matching declaration preset.

## Operator contracts and source alignment

The Clover-specific APIs target the released Clover model equations and state
layout, rather than claiming to be PyTorch standard operators. Preview kernel
provenance and licensing are recorded in `THIRD_PARTY.txt`.

| Operator | Inputs and behavior |
| --- | --- |
| `CloverBlockConv1d` | Packed `[tokens,channels]` BF16/FP16 input, `[slots,channels,3]` clean history, slot indices, left `[channels,1,4]` and learned right `[channels,1,3]` weights, int8 validity, block width up to 32. Produces SiLU output without updating history. |
| `CloverLocalMemory` | Q/K/V `[1,tokens,heads,128]`, FP32 gates and `[slots,value_heads,128,128]` state, slot indices, validity and three per-head scalars. Implements bidirectional block attention plus the decayed clean-history bucket. Does not update state. |
| `CloverRecurrentGdn` | Packed Q/K/V, gates/beta, sequence offsets and initial/final slot vectors. Implements normalized causal delta recurrence with FP32 persistent state; writes sequence output and final state. Input/final slot indices must be valid and final destinations must not race. |
| `CloverCoreset` | Contiguous `[tokens,256]` router logits, block width and routing mass. Keeps the stable top-p support of aggregated per-token top-8 scores and masks other logits. Requires an explicit workspace. |
| `CloverLiveMask` | Current/previous int64 token IDs and the mask ID. Marks rows live if either token is masked, or refresh is forced; updates previous IDs. |
| `CloverMaskRoutes` | Replaces inactive rows' int32 expert IDs with the out-of-range sentinel `num_experts`. |
| `CloverMergeMoe` | Selects fresh routed output for live rows and cached output for other rows; updates the cache. Output and fresh input may alias. |
| `MoeFusedDense` (Hygon) | Existing packed dense-MoE API, with BF16/FP16 weights, top-k weights/IDs and aligned expert metadata. Triton computes gate/up, fused SiLU/down, and weighted reduction. Invalid routes contribute zero; alignment block size 16 is supported and expert biases are unsupported. Requires an explicit workspace. |

The live-row protocol is an optional approximation for iterative diffusion,
adapted from the supplied dllm-h100 TV6 study. It is not a general guarantee that
fixed token IDs imply unchanged hidden states. InfiniLM refreshes every five
previews and always runs full causal replay before committing state.

The standard `MoeAlignBlockSize`, causal convolution, gating, cast, fill,
multiplication, triangular masks and rotary providers also gain Hygon coverage.
They reuse this repository's existing native CUDA-like implementations.

## Validation

Actual BW/DTK runs compare preview operators to independent Torch equations and
verify bitwise state immutability, mixed-length causal state commits, coreset
selection, dense MoE, liveness and cache merging. Public-header and wrapper/JIT
unit tests cover the HIP architecture string and metadata without CUDA scratch
fields. The companion InfiniLM campaign runs full checkpoint generation.

Workspace-bearing operators are invoked through the C++ API with an owning
workspace. The native test adapter in `tests/native/clover_workspace_smoke.cpp`
exercises that path. A missing workspace is rejected before launching a kernel.

Build and run the workspace adapter against the installed SDK:

```sh
python tests/native/build_clover_workspace.py
PYTHONPATH="build/clover-tests:$PYTHONPATH" python tests/native/check_clover_workspace.py
python -m pytest tests/test_clover_preview.py tests/test_clover_live_rows.py -q
```
