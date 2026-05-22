#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter


def main():
    parser = argparse.ArgumentParser(
        description="Compare two InfiniOps pytest operator reports."
    )
    parser.add_argument("left", type=pathlib.Path, help="First report JSON path")
    parser.add_argument("right", type=pathlib.Path, help="Second report JSON path")
    parser.add_argument(
        "--limit",
        type=int,
        default=40,
        help="Max rows to print per diff section (default: 40)",
    )
    args = parser.parse_args()

    left_summary = _load_json(args.left)
    right_summary = _load_json(args.right)
    left_details = _load_details(args.left)
    right_details = _load_details(args.right)

    print(_run_header("left", args.left, left_summary))
    print(_run_header("right", args.right, right_summary))
    print()

    _print_operator_summary_diff(left_summary, right_summary, args.limit)
    print()
    _print_case_diff(left_details, right_details, args.limit)


def _run_header(label, path, summary):
    env = summary.get("environment", {})
    totals = summary.get("totals", {})
    requested = ",".join(env.get("requested_devices") or ["<auto>"])

    return (
        f"{label}: {path}\n"
        f"  requested_devices={requested}\n"
        f"  torch={env.get('torch_version')}\n"
        f"  collected={totals.get('collected')} "
        f"passed={totals.get('passed')} "
        f"skipped={totals.get('skipped')} "
        f"failed={totals.get('failed')}"
    )


def _print_operator_summary_diff(left_summary, right_summary, limit):
    left_ops = {
        _operator_key(row): row for row in left_summary.get("operators", [])
    }
    right_ops = {
        _operator_key(row): row for row in right_summary.get("operators", [])
    }

    left_only = sorted(set(left_ops) - set(right_ops))
    right_only = sorted(set(right_ops) - set(left_ops))
    changed = [
        key
        for key in sorted(set(left_ops) & set(right_ops))
        if _operator_payload(left_ops[key]) != _operator_payload(right_ops[key])
    ]

    print("Operator Diff")
    print(f"  only_left={len(left_only)} only_right={len(right_only)} changed={len(changed)}")

    if left_only:
        print("  only in left:")
        for key in left_only[:limit]:
            print(f"    {key}")

    if right_only:
        print("  only in right:")
        for key in right_only[:limit]:
            print(f"    {key}")

    if changed:
        print("  changed outcomes:")
        for key in changed[:limit]:
            left_row = left_ops[key]
            right_row = right_ops[key]
            print(
                "    "
                f"{key}: "
                f"left={left_row['outcomes']} "
                f"right={right_row['outcomes']}"
            )

            left_reasons = Counter(
                {entry["reason"]: entry["count"] for entry in left_row["skip_reasons"]}
            )
            right_reasons = Counter(
                {entry["reason"]: entry["count"] for entry in right_row["skip_reasons"]}
            )

            if left_reasons != right_reasons:
                print(f"      left_skip_reasons={dict(left_reasons)}")
                print(f"      right_skip_reasons={dict(right_reasons)}")


def _print_case_diff(left_details, right_details, limit):
    left_cases = {_case_key(record): record for record in left_details if record.get("operator")}
    right_cases = {_case_key(record): record for record in right_details if record.get("operator")}

    left_only = sorted(set(left_cases) - set(right_cases))
    right_only = sorted(set(right_cases) - set(left_cases))
    changed = [
        key
        for key in sorted(set(left_cases) & set(right_cases))
        if _case_payload(left_cases[key]) != _case_payload(right_cases[key])
    ]

    print("Case Diff")
    print(f"  only_left={len(left_only)} only_right={len(right_only)} changed={len(changed)}")

    if left_only:
        print("  cases only in left:")
        for key in left_only[:limit]:
            print(f"    {key}")

    if right_only:
        print("  cases only in right:")
        for key in right_only[:limit]:
            print(f"    {key}")

    if changed:
        print("  same case, different result:")
        for key in changed[:limit]:
            left_record = left_cases[key]
            right_record = right_cases[key]
            print(
                "    "
                f"{key}: "
                f"left={left_record['outcome']} "
                f"right={right_record['outcome']}"
            )

            if left_record.get("reason") != right_record.get("reason"):
                print(f"      left_reason={left_record.get('reason')}")
                print(f"      right_reason={right_record.get('reason')}")


def _operator_key(row):
    return f"{row['module']}::{row['operator']}::{row.get('aten_name')}"


def _operator_payload(row):
    return {
        "cases": row["cases"],
        "outcomes": row["outcomes"],
        "skip_reasons": row["skip_reasons"],
        "implementation_indices": row["implementation_indices"],
        "dtypes": row["dtypes"],
    }


def _case_key(record):
    params = {
        key: value
        for key, value in sorted(record.get("params", {}).items())
        if key not in {"device", "rtol", "atol"}
    }
    key = {
        "module": record.get("module"),
        "operator": record.get("operator"),
        "aten_name": record.get("aten_name"),
        "implementation_index": record.get("implementation_index"),
        "params": params,
    }

    return json.dumps(key, sort_keys=True, ensure_ascii=True)


def _case_payload(record):
    return {"outcome": record.get("outcome"), "reason": record.get("reason")}


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_details(summary_path):
    detail_path = summary_path.with_name(f"{summary_path.stem}.details.jsonl")

    if not detail_path.exists():
        return []

    return [
        json.loads(line)
        for line in detail_path.read_text(encoding="utf-8").splitlines()
        if line
    ]


if __name__ == "__main__":
    main()
