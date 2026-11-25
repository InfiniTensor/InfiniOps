import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "a_shape, b_shape, c_shape, a_strides, b_strides, c_strides",
    (
        ((4, 64), (64, 32), (4, 32), None, None, None),
        ((8, 512), (512, 512), (8, 512), None, None, None),
        ((9, 513), (513, 513), (9, 513), None, None, None),
        ((15, 1024), (1024, 512), (15, 512), None, None, None),
        ((8, 512), (512, 512), (8, 512), (1, 8), (1, 512), (1, 8)),
        ((8, 512), (512, 512), (8, 512), (1027, 2), (1027, 2), (1027, 2)),
        ((16, 512), (512, 512), (16, 512), None, None, None),
        ((17, 513), (513, 513), (17, 513), None, None, None),
        ((32, 1024), (1024, 512), (32, 512), None, None, None),
        ((16, 512), (512, 512), (16, 512), (1, 16), (1, 512), (1, 16)),
        ((16, 512), (512, 512), (16, 512), (1027, 2), (1027, 2), (1027, 2)),
        ((1, 257), (257, 65), (1, 65), (1, 1), (1, 257), (1, 1)),
        ((4, 64), (64, 32), (4, 32), (64, 1), None, None),
        ((4, 64), (64, 32), (4, 32), None, (1, 64), None),
        ((4, 64), (64, 32), (4, 32), (64, 1), (1, 64), None),
        ((4, 64), (64, 32), (4, 32), None, None, (64, 1)),
        ((2, 128), (128, 256), (2, 256), None, None, None),
        ((64,), (64,), (), (2,), (2,), None),
        ((64,), (64, 32), (32,), (2,), None, None),
        ((4, 64), (64,), (4,), None, (2,), None),
        ((2, 4, 64), (2, 64, 32), (2, 4, 32), None, None, None),
        ((2, 4, 64), (2, 64, 32), (2, 4, 32), (256, 1, 64), None, None),
        ((4, 8, 128), (4, 128, 64), (4, 8, 64), None, None, None),
        ((2, 1, 4, 64), (3, 64, 32), (2, 3, 4, 32), None, None, None),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-2, 1e-2),
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 1e-2, 1e-2),
    ),
)
def test_matmul(
    a_shape,
    b_shape,
    c_shape,
    a_strides,
    b_strides,
    c_strides,
    dtype,
    device,
    rtol,
    atol,
):
    a = randn_strided(a_shape, a_strides, dtype=dtype, device=device)
    b = randn_strided(b_shape, b_strides, dtype=dtype, device=device)

    c = empty_strided(c_shape, c_strides, dtype=dtype, device=device)

    return Payload(
        _matmul,
        _torch_matmul,
        (a, b, c),
        {},
        rtol=rtol,
        atol=atol,
    )


def _matmul(a, b, c):
    infini.ops.matmul(a, b, c, stream=get_stream(a.device))

    return c


def _torch_matmul(a, b, c):
    result = torch.matmul(a.float(), b.float()).to(c.dtype)
    c.copy_(result)

    return c


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@pytest.mark.parametrize(
    "m, k, n",
    ((1, 257, 65), (8, 512, 512), (9, 513, 513),
     (15, 1024, 512), (16, 512, 512),
     (17, 513, 513), (32, 1024, 512)),
)
@pytest.mark.parametrize("layout", ("row", "col", "padded", "strided", "unaligned"))
def test_matmul_graph_and_rebound_storage(device, dtype, m, k, n, layout):
    if torch.device(device).type != "cuda":
        pytest.skip("requires a CUDA-compatible graph backend")

    def allocate(rows, columns):
        strides = {
            "row": (columns, 1),
            "col": (1, rows),
            "padded": (columns + 7, 1),
            "strided": (columns * 2 + 3, 2),
            "unaligned": (columns, 1),
        }[layout]
        offset = int(layout == "unaligned")
        span = 1 + (rows - 1) * strides[0] + (columns - 1) * strides[1]
        storage = torch.full(
            (span + offset + 8,), 7, dtype=dtype, device=device
        )
        view = storage.as_strided((rows, columns), strides, offset)

        return view, storage

    stream = torch.cuda.Stream(device=device)

    with torch.cuda.stream(stream):
        for _ in range(2):
            a, _ = allocate(m, k)
            b, _ = allocate(k, n)
            y, storage = allocate(m, n)
            a.normal_(0, 0.2)
            b.normal_(0, 0.2)

            def run():
                infini.ops.matmul(a, b, y, stream=get_stream(a.device))

            def check_result():
                reference = torch.matmul(a.float(), b.float()).to(dtype)
                torch.testing.assert_close(
                    y, reference, rtol=1e-2, atol=1e-2
                )
                expected = torch.full_like(storage, 7)
                expected.as_strided(
                    y.shape, y.stride(), y.storage_offset()
                ).copy_(y)
                assert torch.equal(storage, expected)

            run()
            check_result()
            graph = torch.cuda.CUDAGraph()

            with torch.cuda.graph(graph, stream=stream):
                run()

            for replay in range(3):
                if replay == 2:
                    a.zero_()
                    b.zero_()
                else:
                    a.normal_(0, 0.2)
                    b.normal_(0, 0.2)

                graph.replay()
                check_result()

            stream.synchronize()
