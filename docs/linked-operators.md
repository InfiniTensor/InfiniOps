# Linked Operators

The linked backend calls operator symbols exported by an installed third-party
shared library. It applies when a platform package exposes a usable C++ ABI but
does not provide source code or a stable C API.

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
distribution and verifies every required symbol with both `nm` and `readelf`.
Raw `readelf` symbols are demangled with GNU or LLVM `c++filt` before exact
comparison. The resolver writes a CMake manifest and diagnostic JSON under
`generated/linked/`. Generated files are not committed.

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
