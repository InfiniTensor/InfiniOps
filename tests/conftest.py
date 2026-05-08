import hashlib
import random

import pytest
import torch
import torch.utils.benchmark as benchmark

from tests.utils import clone_strided, get_available_devices


def pytest_addoption(parser):
    parser.addoption(
        "--benchmark", action="store_true", help="Run performance benchmarks."
    )
    parser.addoption(
        "--devices",
        nargs="+",
        default=None,
        help="Device(s) to test on (e.g., `--devices ascend cpu`). Accepts platform names (`nvidia`, `metax`, `iluvatar`, `moore`, `cambricon`, `ascend`) or PyTorch device types (`cuda`, `mlu`, `musa`, `npu`). Defaults to all available devices.",
    )


def pytest_configure(config):
    torch.backends.fp32_precision = "tf32"

    config.addinivalue_line(
        "markers",
        "auto_act_and_assert: automatically perform Act and Assert phases using the return values",
    )


def pytest_collectstart(collector):
    if isinstance(collector, pytest.Module):
        _set_random_seed(_hash(collector.name))


@pytest.fixture(scope="module", autouse=True)
def set_seed_per_module(request):
    _set_random_seed(_hash(_module_path_from_request(request)))


@pytest.fixture(autouse=True)
def set_seed_per_test(request):
    _set_random_seed(_hash(_test_case_path_from_request(request)))


@pytest.fixture(scope="module", autouse=True)
def _clear_operator_caches():
    """Clear the C++ operator cache between test modules.

    The `Operator::call()` cache keys on tensor geometry (shape, strides,
    dtype) but not data pointers. When different test modules create tensors
    with identical geometry but different data content (e.g., random
    `cos_sin_cache` tables), a stale cached operator from a prior module
    silently returns wrong results. Clearing the cache at module boundaries
    ensures each module starts with a cold cache.
    """
    yield

    try:
        import infini.ops as ops

        for name in dir(ops):
            cls = getattr(ops, name)

            if hasattr(cls, "clear_cache"):
                cls.clear_cache()
    except ImportError:
        pass


_NPU_UNSUPPORTED_DTYPES = {torch.float64}

# `torch_npu` does not implement random number generation for
# `uint16`/`uint32`/`uint64`.
for _bits in (16, 32, 64):
    _t = getattr(torch, f"uint{_bits}", None)
    if _t is not None:
        _NPU_UNSUPPORTED_DTYPES.add(_t)


@pytest.fixture(autouse=True)
def skip_unsupported_dtypes(request):
    if not hasattr(request.node, "callspec"):
        return

    params = request.node.callspec.params

    if params.get("device") == "npu" and params.get("dtype") in _NPU_UNSUPPORTED_DTYPES:
        pytest.skip(f"{params['dtype']} not supported on Ascend 910B")


@pytest.fixture(autouse=True)
def skip_op_without_platform_impl(request):
    """Skip `device=<torch_type>` parametrizations when the op has no
    implementation on any of the corresponding platforms.

    Only runs for tests that parametrize `device` but not
    `implementation_index` — joint `(device, impl_idx)` parametrize in
    `pytest_generate_tests` already prunes empty-impl pairs at collection
    time, making this check redundant (and wasteful) on those tests.
    """
    if not hasattr(request.node, "callspec"):
        return

    params = request.node.callspec.params

    if "implementation_index" in params:
        return

    device_selectors = _active_device_selectors_for_torch_device(
        request.config, params.get("device")
    )

    if not device_selectors:
        return

    op_cls = _op_class_from_module(request.node.module)

    if op_cls is None or not hasattr(op_cls, "active_implementation_indices"):
        return

    if not any(op_cls.active_implementation_indices(d) for d in device_selectors):
        pytest.skip(
            f"{op_cls.__name__} has no implementation on any "
            f"`{params.get('device')}`-mapped platform/device"
        )


def _set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


_PLATFORM_TO_TORCH_DEVICE = {
    "nvidia": "cuda",
    "metax": "cuda",
    "iluvatar": "cuda",
    "moore": "musa",
    "cambricon": "mlu",
    "ascend": "npu",
}


def _active_device_selectors_for_torch_device(config, torch_device):
    """Return platform or torch device names selected for a torch device type."""
    if not torch_device:
        return ()

    cli_devices = config.getoption("--devices") or ()
    requested_platforms = tuple(
        name
        for name in cli_devices
        if _PLATFORM_TO_TORCH_DEVICE.get(name) == torch_device
    )

    if requested_platforms:
        return requested_platforms

    # The pybind layer maps torch device names (e.g. "cuda") to the backend
    # compiled into the current wheel, avoiding probes of inactive CUDA siblings.
    return (torch_device,)


def _resolve_device(name):
    """Map a platform name (e.g., `ascend`) to a PyTorch device type (e.g., `npu`)."""
    return _PLATFORM_TO_TORCH_DEVICE.get(name, name)


def pytest_generate_tests(metafunc):
    already_parametrized = _get_parametrized_args(metafunc)

    if "dtype" in metafunc.fixturenames and "dtype" not in already_parametrized:
        metafunc.parametrize(
            "dtype, rtol, atol",
            (
                (torch.float32, 1e-7, 1e-7),
                (torch.float16, 1e-3, 1e-3),
                (torch.bfloat16, 1e-3, 1e-3),
            ),
        )

    if "device" in metafunc.fixturenames and "device" not in already_parametrized:
        cli_devices = metafunc.config.getoption("--devices")
        available = get_available_devices()

        if cli_devices:
            devices = tuple(
                d for d in (_resolve_device(x) for x in cli_devices) if d in available
            )
        else:
            devices = ()

        devices = devices or available

        # Joint `(device, implementation_index)` parametrize: generate only
        # pairs where the op has an active implementation on that device.
        # Avoids cross-device pollution — an impl active on `cpu` but not on
        # `npu` no longer appears as a runtime skip in the npu column.
        if (
            "implementation_index" in metafunc.fixturenames
            and "implementation_index" not in already_parametrized
        ):
            op_cls = _op_class_from_module(metafunc.module)

            if op_cls is not None and hasattr(op_cls, "active_implementation_indices"):
                pairs = [
                    (dev, idx)
                    for dev in devices
                    for idx in op_cls.active_implementation_indices(dev)
                ]

                if not pairs:
                    # Emit one skipped placeholder so test IDs read
                    # `[skip-dtype0-...]` instead of `[NOTSET-...]`.
                    # `get_available_devices()` always includes `"cpu"`, so
                    # `devices[0]` is safe.
                    pairs = [
                        pytest.param(
                            devices[0],
                            0,
                            marks=pytest.mark.skip(
                                reason=(
                                    f"{op_cls.__name__} has no active "
                                    "implementation on any available device"
                                )
                            ),
                            id="skip",
                        )
                    ]

                metafunc.parametrize("device, implementation_index", pairs)

                return

        metafunc.parametrize("device", devices)


def _op_class_from_module(module):
    """Derive the `infini.ops.<Op>` class from a `tests/test_<snake>.py` module.

    Test modules have already imported `infini.ops` by the time this runs, so
    no `try/except ImportError` is needed — a real import failure would have
    aborted collection long before.
    """
    module_name = module.__name__.rsplit(".", 1)[-1]

    if not module_name.startswith("test_"):
        return None

    op_snake = module_name[len("test_") :]
    op_pascal = "".join(part.capitalize() for part in op_snake.split("_"))

    import infini.ops as _ops

    return getattr(_ops, op_pascal, None)


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    if pyfuncitem.get_closest_marker("auto_act_and_assert"):
        func_kwargs = {
            arg: pyfuncitem.funcargs[arg] for arg in pyfuncitem._fixtureinfo.argnames
        }

        payload = pyfuncitem.obj(**func_kwargs)

        func = payload.func
        ref = payload.ref
        args = payload.args
        kwargs = payload.kwargs

        ref_args = _clone(args)
        ref_kwargs = _clone(kwargs)

        output = func(*args, **kwargs)
        expected = ref(*ref_args, **ref_kwargs)

        if pyfuncitem.config.getoption("--benchmark"):
            stmt = "func(*args, **kwargs)"

            func_timer = benchmark.Timer(
                stmt=stmt,
                globals={"func": func, "args": args, "kwargs": kwargs},
                label=func.__name__,
                description="InfiniOps",
            )

            ref_timer = benchmark.Timer(
                stmt=stmt,
                globals={"func": ref, "args": ref_args, "kwargs": ref_kwargs},
                label=func.__name__,
                description="Reference",
            )

            func_measurement = func_timer.blocked_autorange()
            ref_measurement = ref_timer.blocked_autorange()

            benchmark.Compare((func_measurement, ref_measurement)).print()

        rtol = payload.rtol
        atol = payload.atol

        assert torch.allclose(output, expected, rtol=rtol, atol=atol)

        return True


def _get_parametrized_args(metafunc):
    parametrized_args = set()

    for marker in metafunc.definition.iter_markers(name="parametrize"):
        args = marker.args[0]

        if isinstance(args, str):
            parametrized_args.update(x.strip() for x in args.split(","))
        elif isinstance(args, (list, tuple)):
            parametrized_args.update(args)

    return parametrized_args


def _test_case_path_from_request(request):
    return f"{_module_path_from_request(request)}::{request.node.name}"


def _module_path_from_request(request):
    return f"{request.module.__name__.replace('.', '/')}.py"


def _hash(string):
    return int(hashlib.sha256(string.encode("utf-8")).hexdigest(), 16) % 2**32


def _clone(obj):
    if isinstance(obj, torch.Tensor):
        return clone_strided(obj)

    if isinstance(obj, tuple):
        return tuple(_clone(a) for a in obj)

    if isinstance(obj, list):
        return [_clone(a) for a in obj]

    if isinstance(obj, dict):
        return {key: _clone(value) for key, value in obj.items()}

    return obj
