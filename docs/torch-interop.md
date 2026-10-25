# External tensor storage wrapping

`ToAtenTensor` converts InfiniRT metadata and delegates nonempty external
storage wrapping to `TorchInterop<device>::WrapExternalTensor`.

The default adapter supplies the known target device to ATen's tensor maker.
ATen therefore does not need to infer the device from an external allocation's
address. Backend initialization uses `lazyInitDevice` when available, including
vendor backports. Older ATen versions initialize the registered backend through
one zero-element factory per device type.

Ascend uses torch_npu's `from_blob` constructor because ordinary ATen storage
does not include the NPU storage descriptor. This specialization is selected
only in Ascend builds; other backend adapters use the common implementation.

The external address denotes logical element zero. Shapes and strides describe
ordinary strided storage. Wrapping does not transfer ownership, and the caller
must keep the allocation alive until all asynchronous uses finish. Zero-element
tensors retain the existing `empty_strided` path, whose data pointer need not
match the supplied empty address.

Storage wrapping does not select a stream or change the existing execution
guards. Stream ordering and native/framework context interoperability remain
the caller's responsibility.

## Validation

The targeted C++ test in `tests/test_cpp_api.py` uses a real InfiniRT CPU
allocation to check shared storage, strides, scalar and empty shapes, and
non-owning destruction. With matching InfiniOps, InfiniRT and PyTorch installs:

```bash
INFINI_OPS_INSTALL_PREFIX=/path/to/infiniops-prefix \
INFINI_RT_ROOT=/path/to/infinirt-prefix \
python -m pytest tests/test_cpp_api.py -k external_tensor_wrapping -q
```

Ascend validation additionally requires a torch_npu-enabled build and the
existing ATen provider tests on an NPU. Changes to stream guards and support
for arbitrary native external streams are outside this storage-wrapping change.
