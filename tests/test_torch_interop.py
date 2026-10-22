"""Contracts shared by generated and handwritten ATen providers."""

import gc
import importlib
import os

import infini.ops
import pytest
import torch

from tests.utils import get_available_devices, get_stream


_BACKENDS = {"cpu": "cpu", "cuda": "nvidia", "mlu": "cambricon", "npu": "ascend"}


def pytest_generate_tests(metafunc):
    if "interop_device" not in metafunc.fixturenames:
        return

    requested = metafunc.config.getoption("--devices") or get_available_devices()
    aliases = {value: key for key, value in _BACKENDS.items()}
    devices = [aliases.get(name, name) for name in requested]
    metafunc.parametrize("interop_device", devices)


@pytest.fixture(scope="module")
def probe():
    try:
        return importlib.import_module("_infini_ops_interop_probe")
    except ModuleNotFoundError:
        if os.environ.get("INFINI_OPS_REQUIRE_INTEROP_TESTS") == "1":
            pytest.fail("Build with INFINI_OPS_BUILD_INTEROP_TESTS=ON.")

        pytest.skip("optional interop probe is not built")


def _stream_pointer(device, stream):
    return getattr(stream, f"{device}_stream")


def test_context_api_compatibility(probe):
    probe.check_context_api_compatibility()


def _host_input(tensor, device):
    if device == "cpu":
        return tensor

    pinned = tensor.pin_memory()
    assert pinned.is_pinned()

    return pinned


@pytest.mark.parametrize(
    "shape,strides", [((3, 7), (7, 1)), ((3, 7), (14, 2)), ((), ()), ((0, 7), (7, 1))]
)
def test_native_allocation_wraps_shape_stride_and_storage(
    probe, interop_device, shape, strides
):
    backend = _BACKENDS[interop_device]
    queue = probe.NativeStream(backend, 0)
    memory = probe.NativeBuffer(backend, 0, 128)
    host = _host_input(torch.arange(128, dtype=torch.float32), interop_device)
    memory.upload(host, queue.pointer)
    queue.synchronize()
    view = memory.view(shape, strides)
    assert tuple(view.shape) == shape
    assert tuple(view.stride()) == strides
    assert view.device.type == interop_device

    if view.numel():
        assert view.data_ptr() == memory.pointer

    torch.testing.assert_close(view.cpu(), host.as_strided(shape, strides))
    del view
    gc.collect()
    copied = torch.empty_like(host, pin_memory=interop_device != "cpu")
    memory.download(copied, queue.pointer)
    queue.synchronize()
    torch.testing.assert_close(copied, host)


def test_wrapped_tensor_keeps_the_test_allocation_owner_alive(probe, interop_device):
    backend = _BACKENDS[interop_device]
    memory = probe.NativeBuffer(backend, 0, 21)
    view = memory.view((3, 7), (7, 1))
    del memory
    gc.collect()
    view.fill_(2.0)
    torch.testing.assert_close(view.cpu(), torch.full((3, 7), 2.0))


@pytest.mark.parametrize("operator", ["exp", "add", "gemm"])
def test_native_stream_orders_upload_provider_and_download(
    probe, interop_device, operator
):
    backend = _BACKENDS[interop_device]
    queue = probe.NativeStream(backend, 0)
    count = 4
    inputs = [probe.NativeBuffer(backend, 0, count) for _ in range(2)]
    output = probe.NativeBuffer(backend, 0, count)
    tensors = [memory.view((2, 2), (2, 1)) for memory in inputs]
    out = output.view((2, 2), (2, 1))
    current = get_stream(interop_device)
    copied = torch.empty(count, pin_memory=interop_device != "cpu")

    for iteration in range(12):
        host = _host_input(
            torch.arange(1, 5, dtype=torch.float32) + iteration, interop_device
        )
        other = _host_input(torch.arange(1, 5, dtype=torch.float32), interop_device)
        inputs[0].upload(host, queue.pointer)
        inputs[1].upload(other, queue.pointer)

        if operator == "exp":
            infini.ops.exp(
                tensors[0], out, implementation_index=8, stream=queue.pointer
            )
            expected = host.exp()
        elif operator == "add":
            infini.ops.add(
                *tensors, 1.0, out, implementation_index=1, stream=queue.pointer
            )
            expected = host + other
        else:
            infini.ops.gemm(*tensors, out, implementation_index=2, stream=queue.pointer)
            expected = (host.reshape(2, 2) @ other.reshape(2, 2)).flatten()

        output.download(copied, queue.pointer)
        queue.synchronize()
        torch.testing.assert_close(copied, expected, rtol=1e-5, atol=1e-5)
        assert get_stream(interop_device) == current


@pytest.mark.parametrize("default", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_context_restores_stream_after_success_and_exception(
    probe, interop_device, default, fails
):
    if interop_device == "cpu":
        if fails:

            def fail():
                raise ValueError("intentional callback failure")

            with pytest.raises(ValueError, match="intentional callback failure"):
                probe.with_context("cpu", 0, 0, fail)
        else:
            assert probe.with_context("cpu", 0, 0, lambda: 17) == 17

        return

    module = getattr(torch, interop_device)
    module.set_device(0)
    outer = module.Stream(device=0)
    queue = probe.NativeStream(_BACKENDS[interop_device], 0)
    requested = 0 if default else queue.pointer
    expected = (
        _stream_pointer(interop_device, module.default_stream(0))
        if default
        else requested
    )

    def callback():
        assert get_stream(interop_device) == expected

        if fails:
            raise ValueError("intentional callback failure")

        return 23

    with module.stream(outer):
        original = get_stream(interop_device)

        if fails:
            with pytest.raises(ValueError, match="intentional callback failure"):
                probe.with_context(_BACKENDS[interop_device], 0, requested, callback)
        else:
            assert (
                probe.with_context(_BACKENDS[interop_device], 0, requested, callback)
                == 23
            )

        assert get_stream(interop_device) == original


@pytest.mark.parametrize("fails", [False, True])
def test_device_restoration_and_stale_runtime_state(probe, interop_device, fails):
    if interop_device == "cpu":
        assert probe.native_device("cpu") == 0

        return

    module = getattr(torch, interop_device)
    backend = _BACKENDS[interop_device]
    target = min(1, module.device_count() - 1)
    queue = probe.NativeStream(backend, target)
    handle = queue.pointer
    module.set_device(0)
    probe.set_native_device(backend, 0)

    def callback():
        assert module.current_device() == target
        assert probe.native_device(backend) == target

        if fails:
            raise ValueError("intentional device callback failure")

    def invoke():
        if fails:
            with pytest.raises(ValueError, match="intentional device callback failure"):
                probe.with_context(backend, target, handle, callback)
        else:
            probe.with_context(backend, target, handle, callback)

    invoke()
    assert probe.native_device(backend) == 0

    if target != 0:
        module.set_device(0)
        probe.set_native_device(backend, target)
        invoke()
        assert probe.native_device(backend) == target
        assert module.current_device() == target

    module.set_device(0)
    del queue
    gc.collect()
    assert probe.native_device(backend) == 0
    assert module.current_device() == 0
    torch.manual_seed(1234)


def test_framework_stream_handle_retains_event_support(probe, interop_device):
    if interop_device == "cpu":
        assert probe.with_context("cpu", 0, 0, lambda: True)

        return

    module = getattr(torch, interop_device)
    outer = module.Stream(device=0)

    def callback():
        event = module.Event()
        event.record(module.current_stream())
        event.synchronize()

    with module.stream(outer):
        # A raw framework handle must retain its event support, including
        # a default stream different from the current stream.
        for stream in [outer, module.default_stream(0)]:
            probe.with_context(
                _BACKENDS[interop_device],
                0,
                _stream_pointer(interop_device, stream),
                callback,
            )
            assert get_stream(interop_device) == _stream_pointer(interop_device, outer)


@pytest.mark.parametrize("operator", ["sdpa", "rotary", "cache"])
def test_other_handwritten_providers_retain_numerics(interop_device, operator):
    # Reuse the existing numerical references with a portable FP32 subset;
    # the full operator suites keep their own backend-specific scope.
    from tests import test_reshape_and_cache as cache
    from tests import test_rotary_embedding as rotary
    from tests import test_scaled_dot_product_attention as sdpa

    if operator == "sdpa":
        for mask in ["none", "bool", "float"]:
            sdpa.test_scaled_dot_product_attention(
                4,
                4,
                mask,
                0.0,
                False,
                None,
                False,
                1,
                torch.float32,
                interop_device,
                1e-5,
                1e-5,
            )
    elif operator == "rotary":
        for has_key in [False, True]:
            rotary.test_rotary_embedding(
                (4,),
                False,
                True,
                0,
                False,
                has_key,
                1,
                torch.float32,
                interop_device,
                1e-5,
                1e-5,
            )
    else:
        for padded in [False, True]:
            cache.test_reshape_and_cache(
                padded,
                "auto",
                1,
                torch.float32,
                interop_device,
                1e-6,
                1e-6,
            )


def test_startup_setting_is_backend_local(probe, interop_device, monkeypatch):
    backend = _BACKENDS[interop_device]
    before = get_stream(interop_device)
    queue = probe.NativeStream(backend, 0)

    for setting in [None, "1", "00"]:
        if setting is None:
            monkeypatch.delenv("TASK_QUEUE_ENABLE", raising=False)
        else:
            monkeypatch.setenv("TASK_QUEUE_ENABLE", setting)

        for stream in [0, queue.pointer]:
            if interop_device == "npu":
                with pytest.raises(RuntimeError, match="TASK_QUEUE_ENABLE=0"):
                    probe.with_context(backend, 0, stream, lambda: None)
            else:
                assert probe.with_context(backend, 0, stream, lambda: 31) == 31

            assert get_stream(interop_device) == before


def test_new_backend_context_needs_no_shared_dispatch_edit(probe):
    assert probe.with_context("fake", 0, 0, lambda: probe.fake_depth()) == 1
    assert probe.fake_depth() == 0

    def fail():
        assert probe.fake_depth() == 1
        raise ValueError("fake backend failure")

    with pytest.raises(ValueError, match="fake backend failure"):
        probe.with_context("fake", 0, 0, fail)

    assert probe.fake_depth() == 0
