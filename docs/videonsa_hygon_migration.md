# VideoNSA on the separated InfiniOps architecture

This migration retains the qualified Hygon/gfx936 kernels from the indicator04
szl-based implementation (InfiniCore commit 772f0c237a2cd9560f6e99e8955d3529f0f68590).
The arithmetic translation units and auxiliary headers under
`src/native/cuda/hygon/ops/nsa_videonsa_prefill/legacy/` are copied without edits.
The old descriptor ABI is replaced by ordinary InfiniOps operator keys.

## Interfaces

- `NsaVideonsaPrefill(q,k,v,g_cmp,g_slc,g_swa,top,block,window,out)` executes
  compression, compression attention, selection, selected attention, window
  attention and gate combination. It includes K/V packing where required.
- `NsaVideonsaSelector(q,k,lse,top,block,scale,out)` returns selector IDs.
- The paged decode keys are `NsaCompressPagedCache` and `NsaPagedAttention`.

Prefill Q is contiguous `[B,T,Hq,128]`; K/V are contiguous `[B,T,Hkv,128]`.
Q/K/V share FP16 or BF16. Gates are contiguous `[B,T,Hq]` and share Q dtype or
FP32; output has Q shape and gate dtype. B is 1..8, T is 1..8192,
Hq is 1..1024, Hkv is 1..32, and Hq/Hkv is an integer at most 32.
Block is a power of two in 2..128; top is 1..min(32,block/2). Window is
nonnegative. Full prefill uses the original D^-1/2 scale.

Selector K is `[B,ceil(T/block),Hkv,128]`; LSE is contiguous FP32 in physical
`[B,Hq,T]` order. Output is contiguous int32 `[B,T,Hkv,top]`. Its supplied scale
must be finite. The original sorting network, tie semantics, causal masks and
low precision rounding are retained.

## Workspace and stream ownership

Use the public `Make` result's `workspace_size_in_bytes()` to reserve a stable
256-byte-aligned device workspace for Prefill. Retain workspace, inputs and
outputs until their stream completes, including every graph replay. A plain
`Handle` has no automatic workspace allocator; either set its workspace/size or
provide a stream-safe lease allocator. Selector needs no workspace. Providers
opt into capture only with this storage lifetime contract. The portable Runner
prepares and retains storage outside capture.

The CMake source property preserves the old `-Xclang
-fno-cuda-host-device-constexpr` option only for these legacy CUDA files; other
operators keep their original compilation flags. The qualified target is gfx936.

## Build and regression

Add `nsa_videonsa_prefill` and `nsa_videonsa_selector` with
`{"implementations":"all"}` to the operator configuration used for a Hygon build.
Build and install the `infiniops` library target with bounded parallelism.

The portable test consumer uses the installed SDK:

```sh
cmake -S tests/videonsa_hygon -B /tmp/videonsa-consumer \
  -DCMAKE_PREFIX_PATH="$INFINI_ROOT" \
  -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"
cmake --build /tmp/videonsa-consumer -j 2
INFINI_VIDEONSA_CONSUMER=/tmp/videonsa-consumer/libprefill_consumer.so \
  python tests/videonsa_hygon/test_frozen.py \
  --stage2-root /path/to/benchmark_test/stage2 \
  --output /path/to/personal-results/frozen.json
```

Run the shared benchmark with `tests/videonsa_hygon` on PYTHONPATH and
`--infini-adapter native_adapter --mode both`. Set
`INFINI_VIDEONSA_CONSUMER` to that consumer library and retain `INFINI_ROOT`.
Numerical references, oracle, payloads, SGLang and acceptance tolerances remain
external and unchanged. Outputs belong in a personal result directory.

This is operator migration. It does not enable a model-default full NSA Prefill
path or establish whole-model performance. Actual model layer tests, frozen
subgraph replay and real-weight model runs must be reported separately.
