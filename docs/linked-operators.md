# Linked Operators

The linked backend calls operator symbols exported by an installed third-party
shared library. Use it when a platform package exposes a usable C++ ABI but does
not provide source code or a stable C API.

Linked implementations live alongside, rather than under, the native and
PyTorch backends:

```text
src/linked/<family>/<device>/
  <library>.yaml
  ops/<operator>/
    binding.yaml
    adapter.h
    adapter.cc
```

The platform library file contains only discovery information:

```yaml
transport: torch
python_distribution: vllm
library_glob: vllm/_C*.so
```

The operator binding names the library and the exact demangled symbols required
by its static adapter:

```yaml
library: vllm
required_symbols:
  - silu_and_mul(at::Tensor&, at::Tensor&)
```

Keep ABI behavior in `adapter.cc`, not in YAML. The adapter owns typed function
declarations, device and stream guards, tensor conversion, synthesized
arguments, layout staging, and return-value copies. This keeps simple adapters
small without restricting more complex provider ABIs.

At configure time, `scripts/resolve_linked_ops.py` locates the installed Python
distribution and verifies every required symbol with both `nm` and `readelf`.
It writes a disposable CMake manifest and diagnostic JSON under
`generated/linked/`; generated files are not committed.

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

Linked implementations use implementation slot `11`; slot `10` is reserved for
Triton. Add one platform/operator adapter at a time and validate the operator's
full layout matrix plus the affected platform smoke set.
