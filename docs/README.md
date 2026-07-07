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
- [Core Types](api/core-types.md): tensors, devices, data types, handles, and
  configuration objects.
- [Operators](api/operators.md): the operator API model, dispatch, caching, and
  implementation layout.
- [Adding ATen-backed operators](aten-operators.md): generated and hand-written
  PyTorch ATen backend guidance.
- [Compatibility](compatibility.md): public API boundary, generated files, and
  internal implementation headers.

Generated API reference pages are intentionally left out of this first
documentation pass. They should be added through a dedicated Doxygen build and
publishing change.
