# Compatibility

This page defines the current documentation and compatibility boundary for
InfiniOps users and contributors.

## Stable User Entries

The intended public entries are:

```cpp
#include <infini/ops.h>
```

```python
import infini.ops
```

The installed Python wheel is the primary user-facing package surface today.
C++ headers are installed for operator development and integration, but the
consumer-facing C++ package boundary should be kept conservative until covered
by dedicated install-consumer tests.

## InfiniRT Dependency

InfiniOps depends on InfiniRT headers and libraries from the configured
`INFINI_RT_ROOT`. Consumers should treat the installed InfiniOps wheel,
InfiniOps headers, and bundled or linked InfiniRT library as a matching set
from the same build.

## Generated Files

The build may generate files under `generated/`, including:

- public operator call instantiations
- generated base classes
- generated PyTorch C++ backend wrappers
- generated Python bindings
- generated metadata for tests

Generated files are build artifacts. Do not edit them by hand. Change the
source generator, allowlist, or source operator definitions instead.

## Internal Headers

Headers under `src/native/**`, `src/torch/**`, and backend-specific runtime
adapters are implementation-facing. They are useful for in-tree development,
tests, and examples, but ordinary users should prefer the public entries above.

## Backend-Specific Behavior

Operator availability and supported dtypes, layouts, strides, and implementation
indexes can differ by backend. Tests should document backend-specific skips or
tolerances explicitly instead of hiding them in broad generated results.

## Source Compatibility

Follow these rules when changing public or semi-public surfaces:

- Keep operator signatures stable unless the PR clearly documents the migration
  path.
- Preserve generated Python binding names when changing implementation details.
- Update documentation and tests in the same PR when user-visible behavior
  changes.
- Treat shared dispatch, wrapper generation, and backend selection changes as
  high-risk and validate them across affected platforms.
