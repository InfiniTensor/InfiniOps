# Linked Operators

The linked backend calls operators provided by an installed third-party shared
library. It supports exact exported C++ symbols and registered PyTorch
Dispatcher operators when a platform package does not provide source code or a
stable C API.

## Source Layout

Linked implementations use the following layout:

```text
src/linked/<transport>/<device>/
  <library>.yaml
  ops/<operator>/
    <implementation>.yaml
    <implementation>.h
    <implementation>.cc
```

The platform library file contains DSO discovery information:

```yaml
python_distribution_package: vllm
library_glob: vllm/_C*.so
```

Files for an operator implementation use the provider name as their common
stem. Multiple implementations for the same operator and device use distinct
file stems and implementation slots.

The implementation YAML identifies the library and the exact demangled symbols
required by its C++ source:

```yaml
library: vllm
required_symbols:
  - silu_and_mul(at::Tensor&, at::Tensor&)
```

A Dispatcher implementation instead declares its exact schema and a required
dispatch key:

```yaml
library: vllm
operator_schema: >-
  _C::gptq_marlin_repack(Tensor b_q_weight, Tensor perm, SymInt size_k,
  SymInt size_n, int num_bits, bool is_a_8bit) -> Tensor
dispatch_key: CUDA
```

When one implementation requires multiple Dispatcher operators from the same
library and dispatch key, `operator_schema` may instead be a non-empty YAML list
of unique, non-empty schema strings. The resolver validates every schema while
force-loading the library once. A single schema is emitted as a scalar string,
including when the input uses a one-item list.

Each implementation uses exactly one contract form. The resolver rejects a
partial Dispatcher contract or a binding that mixes both forms.

## Adapter Boundary

Keep ABI behavior in `<implementation>.cc`, not in YAML. Shared operator
templates own reusable tensor conversion, stream guards, layout staging, and
copy-back behavior. Provider sources own exact typed function declarations,
synthesized arguments, and provider-specific return handling.

For the `torch` transport, an implementation backend inherits its device `C10`
specialization for device identity and external-stream handling, then defines
its provider-specific `Call` ABI.

## Configuration

At configure time, `scripts/resolve_linked_ops.py` locates the installed Python
distribution. It verifies C++ symbols with both `nm` and `readelf`; raw
`readelf` symbols are demangled with GNU or LLVM `c++filt` before exact
comparison. Dispatcher contracts are verified in an isolated Python process by
loading the DSO, comparing the registered schema exactly, and checking the
requested dispatch key. The resolver writes a CMake manifest and diagnostic
JSON under `generated/linked/`. Generated files are not committed.

DSOs that provide Dispatcher registrations are force-loaded only for their own
link item so that the linker cannot discard their static registration code.

Enable the backend independently of generated ATen implementations:

```bash
cmake -S . -B build \
  -DWITH_METAX=ON \
  -DWITH_LINKED=ON \
  -DWITH_TORCH=OFF \
  -DINFINI_OPS_OPS=silu_and_mul
```

The `torch` transport uses the installed PyTorch C++ headers and libraries for
`at::Tensor`, but it does not enable the standard `src/torch` operator backend.
Provider and PyTorch C++ ABIs must match. Configuration fails before compilation
when the distribution, shared library, or an exact required symbol is missing.

InfiniOps does not bundle the provider library. Its resolved directory and the
PyTorch runtime directories are recorded in the installed binary's RPATH, so a
linked build is tied to that Python environment. Reconfigure and rebuild after
moving or replacing the provider environment. In-place changes to a resolved
provider DSO are tracked as CMake configure and link dependencies.

## Implementation Slots

A linked provider declares an implementation slot of `16` or greater in its
`Operator` specialization. The slot must be unique for that operator and
device.
