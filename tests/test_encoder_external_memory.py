import importlib.util
import json
import subprocess
import sys
import sysconfig
from pathlib import Path

import numpy as np
import pybind11
import pytest
import torch

import infini.ops  # noqa: F401 -- load the selected provider before the probe

from tests.test_cpp_api import _compiler, _install_prefixes, _library_dir, _run

pytestmark = pytest.mark.operator_set(
    {"OnnxGather": 8, "Cast": 8, "NativeLayerNorm": 8, "Matmul": 8, "Add": 1}
)


@pytest.fixture(scope="session")
def external_memory_module(tmp_path_factory, request):
    from tests.conftest import missing_operator_requirement, _resolve_device
    from tests.utils import get_available_devices

    selected = request.config.getoption("--devices") or get_available_devices()
    supported = [
        d for d in selected if _resolve_device(d) in {"cpu", "cuda", "mlu", "npu"}
    ]
    if not any(
        missing_operator_requirement(pytestmark.args[0], (d,)) is None
        for d in supported
    ):
        message = "External-memory test requires the retrieval operator set"
        if request.config.getoption("--require-operator-set"):
            pytest.fail(message)
        pytest.skip(message)
    prefixes = _install_prefixes()
    directory = tmp_path_factory.mktemp("encoder_external_memory")
    binary = directory / (
        "_encoder_external_memory" + sysconfig.get_config_var("EXT_SUFFIX")
    )
    libraries = list(
        dict.fromkeys(
            _library_dir(prefixes, name)
            for name in ("libinfiniops.so", "libinfinirt.so")
        )
    )
    _run(
        [
            _compiler("CXX", "c++"),
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-UNDEBUG",
            "-Wno-deprecated-declarations",
            "-I" + pybind11.get_include(),
            "-I" + sysconfig.get_path("include"),
            *("-I" + str(p / "include") for p in prefixes),
            str(Path(__file__).with_name("encoder_external_memory.cc")),
            *("-L" + str(p) for p in libraries),
            "-linfiniops",
            "-linfinirt",
            *("-Wl,-rpath," + str(p) for p in libraries),
            "-o",
            str(binary),
        ]
    )
    spec = importlib.util.spec_from_file_location("_encoder_external_memory", binary)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_encoder_with_infinirt_memory_and_stream(external_memory_module, device):
    # No input or output device allocation is owned by Torch in this path.
    names = {"cpu": "cpu", "cuda": "nvidia", "mlu": "cambricon", "npu": "ascend"}
    if device not in names:
        pytest.skip("External-memory integration covers CPU, CUDA, MLU and NPU")
    accelerator = getattr(torch, device) if device != "cpu" else None
    count = min(2, accelerator.device_count()) if accelerator else 1
    previous = accelerator.current_device() if accelerator else None
    table = (np.arange(32, dtype=np.float32) % 7 - 3).reshape(8, 4)
    indices = np.array([[7, 0, 3], [1, 6, 2]])
    x = 2 * table[indices]
    expected = (x - x.mean(-1, keepdims=True)) / np.sqrt(
        x.var(-1, keepdims=True) + 1e-5
    )
    # Also exercise the entry point in a fresh interpreter without calling any
    # Torch device/stream API or creating a Torch device tensor first.
    code = f"""
import importlib.util, json
import infini.ops
spec = importlib.util.spec_from_file_location('_encoder_external_memory', {external_memory_module.__file__!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print('RAW_RESULT=' + json.dumps(module.run({names[device]!r}, 0)))
"""
    child = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert child.returncode == 0, child.stdout + child.stderr
    line = next(
        line for line in child.stdout.splitlines() if line.startswith("RAW_RESULT=")
    )
    cold = np.array(json.loads(line.removeprefix("RAW_RESULT=")))
    np.testing.assert_allclose(
        cold[:24].reshape(2, 3, 4), expected, atol=1e-5, rtol=1e-4
    )
    np.testing.assert_array_equal(cold[30:].reshape(2, 3, 4), x)
    try:
        for index in range(count):
            first = np.array(external_memory_module.run(names[device], index))
            second = np.array(external_memory_module.run(names[device], index))
            np.testing.assert_allclose(
                first[:24].reshape(2, 3, 4), expected, atol=1e-5, rtol=1e-4
            )
            np.testing.assert_array_equal(
                first[24:30], indices.astype(np.float32).ravel()
            )
            np.testing.assert_array_equal(first[30:].reshape(2, 3, 4), x)
            np.testing.assert_array_equal(first, second)
    finally:
        if accelerator:
            accelerator.set_device(previous)


def test_external_output_cannot_be_silently_reallocated(device):
    from tests.utils import get_stream

    left = torch.ones((2, 3), device=device)
    right = torch.ones((3, 4), device=device)
    output = torch.full((1,), -123.0, device=device)
    address = output.data_ptr()
    with pytest.raises(RuntimeError, match="resiz|size|shape|output"):
        infini.ops.matmul(
            left, right, output, implementation_index=8, stream=get_stream(device)
        )
    assert output.data_ptr() == address
    assert tuple(output.shape) == (1,)
    assert output.item() == -123


def test_cold_npu_initialization_across_python_and_native_threads(
    external_memory_module, device
):
    if device != "npu":
        pytest.skip("Requires Ascend lazy initialization")
    source = f"""
import importlib.util, threading
import infini.ops
spec = importlib.util.spec_from_file_location('_encoder_external_memory', {external_memory_module.__file__!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
barrier = threading.Barrier(2)
errors = []
def native_worker():
    try:
        barrier.wait()
        module.run_without_gil('ascend', 0)
    except BaseException as error:
        errors.append(error)
worker = threading.Thread(target=native_worker)
worker.start()
barrier.wait()
module.run('ascend', 0)
worker.join()
assert not errors, errors
"""
    child = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, timeout=60
    )
    assert child.returncode == 0, child.stdout + child.stderr
