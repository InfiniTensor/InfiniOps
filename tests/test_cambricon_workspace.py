import os
from pathlib import Path

import pytest

from tests.test_cpp_api import _compiler, _install_prefixes, _library_dir, _run


@pytest.mark.operator_set({"Relu": 0})
def test_cambricon_provider_workspace_allocations(device, tmp_path):
    if device != "mlu":
        pytest.skip("Requires the native Cambricon provider and CNRT")
    prefixes = _install_prefixes()
    libraries = list(
        dict.fromkeys(
            _library_dir(prefixes, name)
            for name in ("libinfiniops.so", "libinfinirt.so")
        )
    )
    sdk = Path(os.environ.get("NEUWARE_HOME", "/usr/local/neuware"))
    source = Path(__file__).with_name("cambricon_workspace.cc")
    binary = tmp_path / "cambricon_workspace"
    _run(
        [
            _compiler("CXX", "c++"),
            "-std=c++17",
            "-UNDEBUG",
            "-I" + str(source.parents[1] / "src"),
            "-I" + str(sdk / "include"),
            *("-I" + str(p / "include") for p in prefixes),
            str(source),
            *("-L" + str(p) for p in libraries),
            "-L" + str(sdk / "lib64"),
            "-linfiniops",
            "-linfinirt",
            "-lcnrt",
            "-Wl,--wrap=cnrtMalloc,--wrap=cnrtFree",
            *("-Wl,-rpath," + str(p) for p in libraries),
            "-o",
            str(binary),
        ]
    )
    result = _run([str(binary)])
    print(result.stdout, end="")
    assert "nodes=64 provider_mallocs=0 provider_bytes=0" in result.stdout
    assert "fallback_mallocs=1 fallback_frees=1" in result.stdout
