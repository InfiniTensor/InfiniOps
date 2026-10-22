# External runtime interoperability

`ToAtenTensor` translates InfiniRT metadata and delegates nonempty external
storage wrapping to `TorchInterop<device>::WrapExternalTensor`. The default
implementation supplies an explicit target device to ATen. The Ascend adapter
uses torch_npu's storage descriptor constructor. Platform headers are assembled
only in `src/torch/interop.h`; common conversion and generated call bodies do
not select individual accelerator types.

Initialization detects ATen's shared lazy-init API by capability, including
vendor backports. Older versions initialize their registered backend through
one zero-element factory per device type. This avoids a hardware-specific
initialization list and does not allocate an extra tensor for each wrapping.

The external address is the logical element-zero pointer. Shapes and strides
describe ordinary strided storage. This interface does not import opaque
vendor formats or transfer ownership. The caller must retain the allocation
until all asynchronous uses finish. Zero-element tensors retain the existing
`empty_strided` path, so their returned data pointer is not promised to match
the supplied empty address.

`WithTorchExecutionContext` wraps both tensor conversion and computation. Its
default implementation uses the backend's existing `C10` stream guard. CPU has
an explicit no-op adapter; unsupported accelerator contexts do not silently
execute without a guard. Generated providers, Add, Gemm, SDPA, rotary embedding,
reshape-and-cache and the existing FlashAttention execution wrappers use this
same entry point. Instantiations requiring SDK guards are compiled only for
enabled backends.

NVIDIA, Cambricon and Ascend explicitly map a null handle to their framework's
device default stream. This is not the currently selected PyTorch stream.
Cambricon's default queue can be a real non-null CNRT queue: null is an adapter
convention, not a promise that native CNRT APIs accept a null queue. Use the
same explicit InfiniRT-created stream for native async copies and ATen calls.
Cross-stream dependencies remain the caller's responsibility.

Before Ascend enters a guard, it aligns torch_npu's cached device with ACL's
actual entry device, not the requested target device. The guard then selects
the target and restores the actual entry device on normal or exceptional exit.
This also repairs stale TLS after a native runtime device switch. The ACL
entry device must already be initialized by the external runtime. Set
`TASK_QUEUE_ENABLE=0` before importing torch_npu. The adapter rejects other
values before changing context, for both default and explicit streams. Reading
the environment cannot verify when the setting took effect; changing it after
torch_npu initialization is unsupported.

The Ascend adapter preserves a current or default framework stream when its
raw handle is passed back through the API. Rewrapping that same address as an
external stream loses framework identity and can break event recording.
Arbitrary external streams still have the SDK's external-stream restrictions:
torch_npu 2.7.1.post10 rejects `NPUEvent::record` on them, including operators
that internally record such events. This adapter does not silently move that
work to a managed stream or synchronize the entire device to simulate support.
Use a supported external-stream operator, or the current/default framework
stream for an operator requiring framework events. This is not a claim that
every ATen operator works on every native stream.

Adding an accelerator normally requires its `C10` specialization and one
assembly include. Override `TorchInterop` only when default external storage
construction is insufficient, inheriting the shared context implementation.
There is no second accelerator dispatch list in the generated call body.
Build-time device registration and explicit template instantiation still list
the supported targets.

This change does not alter GELU output policy, add encoder operators, promise
arbitrary output layouts, or advertise graph capture safety. Allocator and
workspace lifetime during capture require separate validation. In particular,
default runtime state restoration does not provide inter-thread scheduling or
allow incompatible PrivateUse1 extensions to coexist in one process.

## Contract tests

Configure a Torch-enabled build with Python bindings and
`-DINFINI_OPS_BUILD_INTEROP_TESTS=ON`. The private
`_infini_ops_interop_probe` module is built in `build/src` and is not installed
or exposed as production API. Put that directory on `PYTHONPATH`, along with
the installed `infini.ops` package, then run:

```bash
INFINI_OPS_REQUIRE_INTEROP_TESTS=1 python -m pytest tests/test_torch_interop.py --devices cuda -q -rs
```

Use `cpu cuda`, `mlu`, or `npu` for the corresponding builds. The required
environment flag turns a missing probe into a failure; ordinary builds may
skip this optional test module. The tests use real InfiniRT allocations and
streams, native asynchronous upload/download around generated Exp and
handwritten Add/Gemm, strided/scalar/empty views, non-owning storage, callback
exceptions, default/non-default streams and available non-default devices.
The test wrapper explicitly keeps its allocation owner alive; production
callers remain responsible for their own lifetime management. CPU copies are
synchronous because its runtime does not implement async memcpy. A fake C10
specialization verifies context extension without editing common dispatch.
