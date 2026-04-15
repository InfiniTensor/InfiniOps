"""Performance benchmark orchestrator for RMSNorm using msprof."""

import csv
import glob
import json
import os
import subprocess
import sys


CASES_FILE = os.path.join(os.path.dirname(__file__), "rms_norm_cases.jsonl")
RUNNER_SCRIPT = os.path.join(os.path.dirname(__file__), "run_rms_norm_case.py")
MSPROF_BASE = "/tmp/msprof_rms_norm"

# OP Type keyword for filtering in op_summary CSV.
OP_TYPE_KEYWORD = "rms_norm"


def load_cases():
    cases = []
    with open(CASES_FILE) as f:
        for line in f:
            line = line.strip()

            if line:
                cases.append(json.loads(line))

    return cases


def run_msprof(case, output_dir, iters=20, warmup=10):
    """Run a single case under msprof profiling."""
    # Write a self-contained wrapper to avoid shell quoting issues.
    os.makedirs(os.path.dirname(output_dir + "_") or ".", exist_ok=True)
    wrapper = output_dir + "_run.py"

    with open(wrapper, "w") as f:
        f.write(
            "import json, torch, torch_npu, ascend_kernel\n"
            f"case = {json.dumps(case)}\n"
            "shape = tuple(case['shape'])\n"
            "dtype = getattr(torch, case['dtype'])\n"
            "eps = case['eps']\n"
            "hidden_dim = shape[-1]\n"
            "x = torch.randn(shape, dtype=dtype, device='npu')\n"
            "w = torch.randn(hidden_dim, dtype=dtype, device='npu')\n"
            f"for _ in range({warmup}):\n"
            "    _ = torch.ops.npu.rms_norm(x, w, eps)\n"
            "torch.npu.synchronize()\n"
            f"for _ in range({iters - warmup}):\n"
            "    _ = torch.ops.npu.rms_norm(x, w, eps)\n"
            "torch.npu.synchronize()\n"
        )

    cmd = (
        f"msprof --output={output_dir} --task-time=l1 --runtime-api=on "
        f'--application="python3 {wrapper}"'
    )
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    try:
        os.remove(wrapper)
    except OSError:
        pass

    if result.returncode != 0:
        print(f"  msprof FAILED for case {case['id']}: {result.stderr[-300:]}")

        return False

    return True


def parse_op_summary(output_dir, op_type_keyword):
    """Parse msprof op_summary CSV for the target OP Type."""
    # Find the op_summary CSV.
    pattern = os.path.join(output_dir, "**", "op_summary_*.csv")
    csv_files = glob.glob(pattern, recursive=True)

    if not csv_files:
        return None

    csv_file = csv_files[0]
    results = []

    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            op_type = row.get("OP Type", "")

            if op_type_keyword.lower() in op_type.lower():
                results.append(row)

    return results


def main():
    cases = load_cases()
    print(f"Loaded {len(cases)} benchmark cases")
    print("=" * 80)

    all_results = []

    for case in cases:
        case_id = case["id"]
        desc = case["desc"]
        output_dir = os.path.join(MSPROF_BASE, f"case_{case_id}")
        print(f"[Case {case_id}] {desc} shape={case['shape']} dtype={case['dtype']}")

        ok = run_msprof(case, output_dir, iters=20, warmup=10)

        if not ok:
            all_results.append({
                "id": case_id,
                "desc": desc,
                "shape": str(case["shape"]),
                "dtype": case["dtype"],
                "status": "FAILED",
            })
            continue

        rows = parse_op_summary(output_dir, OP_TYPE_KEYWORD)

        if not rows:
            print(f"  WARNING: No matching OP Type '{OP_TYPE_KEYWORD}' found")
            all_results.append({
                "id": case_id,
                "desc": desc,
                "shape": str(case["shape"]),
                "dtype": case["dtype"],
                "status": "NO_MATCH",
            })
            continue

        # Aggregate Task Duration across matching rows.
        durations = []

        for row in rows:
            dur = row.get("Task Duration(us)", "0")

            try:
                durations.append(float(dur))
            except ValueError:
                pass

        if durations:
            avg_dur = sum(durations) / len(durations)
            min_dur = min(durations)
            max_dur = max(durations)
        else:
            avg_dur = min_dur = max_dur = 0.0

        print(f"  Task Duration: avg={avg_dur:.2f}us min={min_dur:.2f}us max={max_dur:.2f}us ({len(durations)} calls)")

        result = {
            "id": case_id,
            "desc": desc,
            "shape": str(case["shape"]),
            "dtype": case["dtype"],
            "status": "OK",
            "avg_duration_us": avg_dur,
            "min_duration_us": min_dur,
            "max_duration_us": max_dur,
            "num_calls": len(durations),
        }

        # Extract additional hardware metrics if available.
        if rows:
            for key in ["Task Wait Time(us)", "Block Dim"]:
                val = rows[0].get(key, "")

                if val:
                    result[key] = val

        all_results.append(result)

    # Save JSON.
    json_path = os.path.join(os.path.dirname(__file__), "rms_norm_perf.json")

    with open(json_path, "w") as f:
        json.dump({"results": all_results}, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"JSON results saved to: {json_path}")

    # Print summary table.
    print(f"\n{'ID':>3} {'Shape':>20} {'Dtype':>8} {'Avg(us)':>10} {'Min(us)':>10} {'Max(us)':>10} {'Calls':>6}")
    print("-" * 75)

    for r in all_results:
        if r["status"] == "OK":
            print(
                f"{r['id']:>3} {r['shape']:>20} {r['dtype']:>8} "
                f"{r['avg_duration_us']:>10.2f} {r['min_duration_us']:>10.2f} "
                f"{r['max_duration_us']:>10.2f} {r['num_calls']:>6}"
            )
        else:
            print(f"{r['id']:>3} {r['shape']:>20} {r['dtype']:>8}   {r['status']}")


if __name__ == "__main__":
    main()
