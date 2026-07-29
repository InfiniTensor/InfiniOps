import importlib.util
import json
from pathlib import Path

import torch
import triton


_JIT_DIR = Path(__file__).resolve().parent
_OPS_DIR = _JIT_DIR.parent / "ops"


def _do_compile(
    op_name,
    out_prefix,
    num_warps,
    num_stages,
    device_id,
    signature,
):

    source_path = _OPS_DIR / f"{op_name}/{op_name}.py"
    spec = importlib.util.spec_from_file_location(source_path.stem, source_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "kernel")
    while not isinstance(fn, triton.runtime.JITFunction):
        fn = fn.fn

    sig_parts = [p.strip() for p in signature.split(",")] if signature else []
    assert len(sig_parts) == len(fn.arg_names), (
        f"signature length {len(sig_parts)} != kernel param count {len(fn.arg_names)}"
    )

    sig_dict = {}
    const_dict = {}
    attr_dict = {}

    constexprs = {}
    for part in sig_parts:
        if "=" in part:
            name, val = part.split("=", 1)
            constexprs[name.strip()] = int(val)

    for i, (name, param, part) in enumerate(zip(fn.arg_names, fn.params, sig_parts)):
        if param.is_constexpr:
            const_dict[(i,)] = constexprs[name]
            sig_dict[name] = "constexpr"
        elif part.endswith(":1"):
            const_dict[(i,)] = 1
            sig_dict[name] = "constexpr"
        elif part.endswith(":16"):
            sig_dict[name] = part[:-3]
            attr_dict[(i,)] = [["tt.divisibility", 16]]
        else:
            sig_dict[name] = part

    src = triton.compiler.ASTSource(
        fn=fn, signature=sig_dict, constexprs=const_dict, attrs=attr_dict
    )

    with torch.cuda.device(device_id):
        target = triton.runtime.driver.active.get_current_target()
        ccinfo = triton.compile(
            src,
            target=target,
            options={"num_warps": num_warps, "num_stages": num_stages},
        )

    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    backend = triton.compiler.make_backend(target)
    bin_ext = backend.binary_ext
    cubin = ccinfo.asm[bin_ext]
    with open(out_prefix + ".cubin", "wb") as f:
        f.write(cubin)

    meta = {
        "name": getattr(ccinfo.metadata, "name", fn.__name__),
        "shared": getattr(ccinfo.metadata, "shared", 0),
        "num_warps": getattr(ccinfo.metadata, "num_warps", num_warps),
        "arch": target.arch if hasattr(target, "arch") else 80,
        "global_scratch_size": getattr(ccinfo.metadata, "global_scratch_size", 0),
        "profile_scratch_size": getattr(ccinfo.metadata, "profile_scratch_size", 0),
        "op_name": op_name,
        "signature": signature,
    }
    with open(out_prefix + ".json", "w") as f:
        json.dump(meta, f)


def _load_kernel_fn(op_name):
    source_path = _OPS_DIR / f"{op_name}/{op_name}.py"
    spec = importlib.util.spec_from_file_location(source_path.stem, source_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "kernel")
    while not isinstance(fn, triton.runtime.JITFunction):
        fn = fn.fn
    return fn


def _do_autotune(op_name, configs, args, grids, warmup, rep, device_id):
    fn = _load_kernel_fn(op_name)
    best_idx = 0
    best_time = float("inf")
    with torch.cuda.device(device_id):
        for i, cand in enumerate(configs):
            constexprs = {kv[0]: kv[1] for kv in cand["constexprs"]}
            num_warps = cand["num_warps"]
            num_stages = cand["num_stages"]
            grid = tuple(grids[i])

            out_prefix = cand["out_prefix"]
            _do_compile(
                op_name, out_prefix, num_warps, num_stages, device_id, cand["full_sig"]
            )

            def _kernel_call(
                g=grid, a=args, ce=constexprs, nw=num_warps, ns=num_stages
            ):
                fn[g](*a, **ce, num_warps=nw, num_stages=ns)

            try:
                t = triton.testing.do_bench(
                    _kernel_call, warmup=warmup, rep=rep, quantiles=(0.5, 0.2, 0.8)
                )[0]
                if t < best_time:
                    best_time = t
                    best_idx = i
            except Exception:
                pass
    return best_idx
