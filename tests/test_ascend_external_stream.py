"""Ascend-only regressions for the external-runtime execution boundary."""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

pytest.importorskip("torch_npu")
import infini.ops

pytestmark = pytest.mark.smoke


def test_external_stream_restores_callers_device():
    original_device = torch.npu.current_device()
    original_stream = torch.npu.current_stream().npu_stream
    try:
        for index in range(min(2, torch.npu.device_count())):
            with torch.npu.device(index):
                stream = torch.npu.Stream()
                x = torch.arange(6, device=f"npu:{index}", dtype=torch.float32)
                out = torch.empty_like(x, dtype=torch.float32)
                stream.wait_stream(torch.npu.current_stream())
            torch.npu.set_device(original_device)
            infini.ops.exp(x, out, implementation_index=8, stream=stream.npu_stream)
            stream.synchronize()
            assert torch.npu.current_device() == original_device
            assert torch.npu.current_stream().npu_stream == original_stream
            torch.testing.assert_close(out.cpu(), x.cpu().exp())
    finally:
        torch.npu.set_device(original_device)


def test_changing_environment_does_not_hide_an_enabled_task_queue():
    source = textwrap.dedent(r"""
        import ctypes
        import os
        from pathlib import Path
        import torch
        import torch_npu
        import infini.ops
        library = ctypes.CDLL(str(Path(torch_npu.__file__).parent / 'lib/libtorch_npu.so'))
        name = 'OpApiGetTaskQueueEnable'
        query = getattr(library, '_Z' + str(len(name)) + name + 'v')
        query.restype = ctypes.c_uint32
        assert query() != 0
        os.environ['TASK_QUEUE_ENABLE'] = '0'
        assert query() != 0, 'backend queue state must remain cached'
        x = torch.ones((2, 3), dtype=torch.float32, device='npu')
        out = torch.full((2, 3), -123.0, device='npu')
        for stream in (0, torch.npu.current_stream().npu_stream):
            try:
                infini.ops.exp(x, out, implementation_index=8, stream=stream)
            except RuntimeError as error:
                assert 'TASK_QUEUE_ENABLE=0 before' in str(error), str(error)
            else:
                raise AssertionError('An initialized host task queue was accepted')
        assert torch.all(out.cpu() == -123)
        print('Cached task queue rejected before external stream execution')
    """)
    environment = os.environ.copy()
    environment["TASK_QUEUE_ENABLE"] = "1"
    environment.pop("ASCEND_LAUNCH_BLOCKING", None)
    result = subprocess.run(
        [sys.executable, "-c", source],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
