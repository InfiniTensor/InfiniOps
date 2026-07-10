# InfiniOps Documentation

InfiniOps is a high-performance operator library for InfiniCore. It provides
Python bindings and C++ operator implementations across CPU, NVIDIA, Iluvatar,
Hygon, MetaX, Cambricon, Moore, Ascend, and other backends.

The common public include entry is:

```cpp
#include <infini/ops.h>
```

The Python package entry is:

```python
import infini.ops
```

Start with these pages:

- [Getting Started](getting-started.md): install InfiniRT, build InfiniOps, and
  run a minimal operator call.
- [Build and Test](build.md): build options, smoke builds, and test commands.
- [Backends](backends.md): supported backend options and backend-specific
  requirements.
- [Examples](examples.md): Python and C++ examples currently available in the
  repository.
- [Core Types](api/core-types.md): tensors, devices, data types, handles, and
  configuration objects.
- [Operators](api/operators.md): the operator API model, dispatch, caching, and
  implementation layout.
- [API Reference](api/reference.md): generate the Doxygen C++ API reference.
- [Adding ATen-backed operators](aten-operators.md): generated and hand-written
  PyTorch ATen backend guidance.
- [Compatibility](compatibility.md): public API boundary, generated files, and
  internal implementation headers.

The API reference can be generated locally with Doxygen. Publishing the
generated HTML is intentionally handled separately from the local documentation
target.
