"""Exercise the installed legacy Python entry without source-tree imports."""

import json
from pathlib import Path
import sys

import infini.ops
import torch


prefix = Path(sys.argv[1]).resolve()
module = Path(infini.ops.__file__).resolve()

if not module.is_relative_to(prefix):
    raise RuntimeError(f"Unexpected installed module `{module}`.")

x = torch.tensor([1.0, -2.0, 3.0])
y = torch.tensor([4.0, 5.0, -6.0])
out = torch.full_like(x, float("nan"))
infini.ops.add(x, y, out)
torch.testing.assert_close(out, torch.tensor([5.0, 3.0, -3.0]), rtol=0, atol=0)
infini.ops.add(x, y, 2.0, out)
torch.testing.assert_close(out, torch.tensor([9.0, 8.0, -9.0]), rtol=0, atol=0)
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
c = torch.full_like(a, float("nan"))
infini.ops.gemm(a, b, c)
torch.testing.assert_close(
    c, torch.tensor([[19.0, 22.0], [43.0, 50.0]]), rtol=0, atol=0
)
print(
    json.dumps(
        {
            "module": str(module),
            "torch": torch.__version__,
            "exports": sorted(
                name for name in dir(infini.ops) if not name.startswith("_")
            ),
            "checks": ["add.default", "add.alpha", "gemm.default"],
            "loaded_libraries": sorted(
                {
                    line.split()[-1]
                    for line in Path("/proc/self/maps").read_text().splitlines()
                    if "libinfiniops" in line or "libinfinirt" in line
                }
            ),
        }
    )
)
