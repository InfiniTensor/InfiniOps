# Hygon Clover operators

These operators support the companion InfiniLM Clover implementation. Model
forward stays in C++; Hygon driver modules launch warmed Triton binaries.
Compilation uses the installed DTK Triton HIP target (for example `gfx936`,
wave size 64), not CUDA compute-capability arithmetic. On `gfx936` the fused
MoE expert computation runs native kernels instead; see
[Native fused MoE on gfx936](#native-fused-moe-on-gfx936).

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
| `MoeFusedDense` (Hygon) | Existing packed dense-MoE API, with BF16/FP16 weights, top-k weights/IDs and aligned expert metadata: gate/up, fused SiLU/down, and weighted reduction. Invalid routes contribute zero; alignment block size 16 is supported and expert biases are unsupported. Requires an explicit workspace. Native kernels on `gfx936`, Triton otherwise. |

The live-row protocol is an optional approximation for iterative diffusion,
adapted from the supplied dllm-h100 TV6 study. It is not a general guarantee that
fixed token IDs imply unchanged hidden states. InfiniLM refreshes every five
previews and always runs full causal replay before committing state.

The standard `MoeAlignBlockSize`, causal convolution, gating, cast, fill,
multiplication, triangular masks and rotary providers also gain Hygon coverage.
They reuse this repository's existing native CUDA-like implementations.

## Native fused MoE on gfx936

`MoeFusedDense` streams the weights of the active experts once per MoE layer.
On `gfx936` the Hygon provider (`native/cuda/hygon/ops/moe_fused_dense`) runs
three native kernels:

- gate and up projections of 16 features, fused with SiLU into the activation,
  the only intermediate rounded to the element type;
- the down projection, scaled by the routing weight and kept in FP32;
- the weighted sum over the routes of a token.

One wavefront (one workgroup) streams two 16-row weight bands of an expert
block. Each load instruction reads 256 contiguous bytes per row, a step goes
through LDS into the register layout of `v_mmac_f32_16x16x16_{bf16,f16}`, and
about 8 KB of loads stay in flight per wavefront, carried across work items.
The Triton kernels use the same matrix instruction, but compute one 16 x 64
output tile per four-wavefront workgroup with 64-wide k steps staged behind
barriers, which leaves too few weights in flight for a memory-bound step.

What this gains depends on the routing. With the routes of a forward spread
over the experts, every MoE layer streams hundreds of expert blocks and the
native kernels sustain about 1 TB/s of weight traffic. Clover-35B per-rank
shapes (256 experts, hidden 2048, intermediate 256, top-8), routes spread:

| Tokens | Native | Triton | Speedup |
| ---: | ---: | ---: | ---: |
| 32 | 0.51 ms | 1.78 ms | 3.5x |
| 64 | 0.69 ms | 2.98 ms | 4.3x |
| 128 | 0.79 ms | 3.31 ms | 4.2x |
| 256 | 0.82 ms | 3.47 ms | 4.2x |
| 1024 | 1.64 ms | 4.09 ms | 2.5x |

The block coreset of the model concentrates the routes of a 32-token block on a
few experts instead. So few blocks are left to stream that both
implementations are bound by the bytes of those blocks, the activations
re-read per tile included, and take the same time: with about 25 active
blocks, 0.10 ms at 32 tokens for either, the native kernels within 5% of the
Triton ones (`bench_moe_fused_dense_hygon.py --hot 0.8`). In the 32-token
blocks of the model the fused MoE is 4-5% of a forward's GPU kernel time, and
the native kernels save 1-2% end to end.

The kernels are compiled when `HYGON_ARCH` is `gfx936` and take block size 16
with `h` and `i` multiples of 8 of at least 32. Other shapes, and builds for
other Hygon architectures, run the Triton JIT kernels through the same
provider. `INFINI_OPS_HYGON_MOE_TRITON=1` keeps the Triton kernels on `gfx936`
as well.

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
export PYTHONPATH="build/clover-tests:$PYTHONPATH"
python tests/native/check_clover_workspace.py
python tests/native/check_moe_fused_dense_hygon.py
python tests/native/bench_moe_fused_dense_hygon.py
python -m pytest tests/test_clover_preview.py tests/test_clover_live_rows.py -q
```

`check_moe_fused_dense_hygon.py` compares the fused MoE against float32 Torch
references on its native and Triton paths, each with its own rounding, for
spread and concentrated routing. `bench_moe_fused_dense_hygon.py` times the two;
`--hot` concentrates the routes on a few experts.
