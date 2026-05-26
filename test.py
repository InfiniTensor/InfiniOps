import torch
import torch_mlu
import time
import infini.ops

N = 100
WARMUP = 10
device = "mlu"

x = torch.randn(8192, 8192, dtype=torch.float16, device=device)
out = torch.empty_like(x)


def bench(name, fn):
    for _ in range(WARMUP):
        fn()
    torch.mlu.synchronize()
    total = 0.0
    for _ in range(N):
        torch.mlu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.mlu.synchronize()
        total += time.perf_counter() - t0
    print(f"{name:40s} {total/N*1e6:.1f} us")


# 1. PyTorch 原生
bench("PyTorch torch.abs", lambda: torch.abs(x, out=out))

# 2. InfiniOps
stream = torch.mlu.current_stream().mlu_stream
bench("InfiniOps abs", lambda: infini.ops.abs(x, out, stream=stream, implementation_index=8))

# 3. as_strided 包装开销测试：模拟 InfiniOps 内部 ToAtenTensor
shape = x.shape
strides = x.stride()


def as_strided_abs():
    # as_strided 类似 from_blob，创建 view 不拷贝数据
    x_wrap = torch.as_strided(x, shape, strides)
    o_wrap = torch.as_strided(out, shape, strides)
    torch.abs(x_wrap, out=o_wrap)


bench("PyTorch + as_strided abs", as_strided_abs)

# 4. 只测 as_strided 创建开销
def as_strided_only():
    torch.as_strided(x, shape, strides)
    torch.as_strided(out, shape, strides)

bench("as_strided only (no compute)", as_strided_only)

# 5. InfiniOps 不传 stream
bench("InfiniOps abs (no stream)", lambda: infini.ops.abs(x, out, implementation_index=8))

# ========== bitwise_and (binary op) ==========
xi = torch.randint(0, 100, (8192, 8192), dtype=torch.int32, device=device)
yi = torch.randint(0, 100, (8192, 8192), dtype=torch.int32, device=device)
out_bin = torch.empty_like(xi)
shape_i = xi.shape
strides_i = xi.stride()

bench("PyTorch bitwise_and", lambda: torch.bitwise_and(xi, yi, out=out_bin))
bench("InfiniOps bitwise_and", lambda: infini.ops.bitwise_and(xi, yi, out_bin, stream=stream, implementation_index=8))

def as_strided_bitwise_and():
    xi_wrap = torch.as_strided(xi, shape_i, strides_i)
    yi_wrap = torch.as_strided(yi, shape_i, strides_i)
    oi_wrap = torch.as_strided(out_bin, shape_i, strides_i)
    torch.bitwise_and(xi_wrap, yi_wrap, out=oi_wrap)

bench("PyTorch + as_strided bitwise_and", as_strided_bitwise_and)

def as_strided_only_bin():
    torch.as_strided(xi, shape_i, strides_i)
    torch.as_strided(yi, shape_i, strides_i)
    torch.as_strided(out_bin, shape_i, strides_i)

bench("as_strided only binary (no compute)", as_strided_only_bin)

bench("InfiniOps bitwise_and (no stream)", lambda: infini.ops.bitwise_and(xi, yi, out_bin, implementation_index=8))
