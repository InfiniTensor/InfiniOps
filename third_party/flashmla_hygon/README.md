# Hygon FlashMLA kernel subset

This directory pins the include closure for the gfx93 MODEL1 sparse FP8 decode
instantiation and the gfx9 scheduling/combination kernels. InfiniOps builds them
with DTK AICC; no Python extension or PyTorch runtime is required.

The original file hashes and copied file hashes are in sources.json. Sources
were imported from the existing flashmla-hygon checkout on the development host.
Only the required kernel/header closure is included. References to ATen scalar
macros are replaced by their equivalent definitions; the pinned kernel arithmetic
is unchanged.

FlashMLA and CUTLASS licenses are preserved in LICENSE and CUTLASS_LICENSE.txt.
The InfiniOps runtime bridge lives in src/native/cuda/hygon/flash_mla relative to
the InfiniOps root. The default integration is enabled for gfx936/gfx938.
