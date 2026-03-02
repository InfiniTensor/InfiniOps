import hashlib
import random

import pytest
import torch

from tests.utils import get_available_devices


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


def _set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


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
        metafunc.parametrize("device", get_available_devices())


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
        return obj.clone()

    if isinstance(obj, tuple):
        return tuple(_clone(a) for a in obj)

    if isinstance(obj, list):
        return [_clone(a) for a in obj]

    if isinstance(obj, dict):
        return {key: _clone(value) for key, value in obj.items()}

    return obj
