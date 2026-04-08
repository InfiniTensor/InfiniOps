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
        help="Device(s) to test on (e.g., --devices ascend cpu). Accepts platform names (ascend, nvidia, cambricon, metax, moore, iluvatar) or PyTorch device types (npu, cuda, mlu, musa). Defaults to all available devices.",
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


_NPU_UNSUPPORTED_DTYPES = {torch.float64}

# torch_npu does not implement random number generation for uint16/uint32/uint64.
for _bits in (16, 32, 64):
    _t = getattr(torch, f"uint{_bits}", None)
    if _t is not None:
        _NPU_UNSUPPORTED_DTYPES.add(_t)


@pytest.fixture(autouse=True)
def skip_unsupported_dtype(request):
    if not hasattr(request.node, "callspec"):
        return
    params = request.node.callspec.params

    if params.get("device") == "npu" and params.get("dtype") in _NPU_UNSUPPORTED_DTYPES:
        pytest.skip(f"{params['dtype']} not supported on Ascend 910B")


def _set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


_PLATFORM_TO_TORCH_DEVICE = {
    "nvidia": "cuda",
    "iluvatar": "cuda",
    "metax": "cuda",
    "cambricon": "mlu",
    "moore": "musa",
    "ascend": "npu",
}


def _resolve_device(name):
    """Map a platform name (e.g., ``ascend``) to a PyTorch device type (e.g., ``npu``)."""
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

        metafunc.parametrize("device", devices or available)


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
