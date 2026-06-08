#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter, defaultdict

from operator_categories import (
    CATEGORIES,
    inventory_operator_name,
    load_operator_inventory,
)


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

_DISPLAY_NAMES = {
    "ascend": "Ascend",
    "cambricon": "Cambricon",
    "cpu": "CPU",
    "hygon": "Hygon",
    "iluvatar": "Iluvatar",
    "metax": "MetaX",
    "moore": "Moore",
    "nvidia": "Nvidia",
}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Print a compact summary for one or more pytest operator reports, "
            "including both case-level skips and operator-level skip-only counts."
        )
    )
    parser.add_argument(
        "inputs", nargs="+", type=pathlib.Path, help="Report JSON path(s)"
    )
    parser.add_argument(
        "--show-skip-only",
        action="store_true",
        help="Also list operator names whose status is SKIPPED_ONLY.",
    )
    args = parser.parse_args()

    inventory = _load_inventory()
    first = True

    for path in args.inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))

        for label, summary in _extract_summaries(path, payload):
            if not first:
                print("")
            first = False
            _print_summary(label, summary, inventory, args.show_skip_only)


def _load_inventory():
    return load_operator_inventory()


def _extract_summaries(path, payload):
    if {"left", "right", "operator_diff", "case_diff"} <= set(payload):
        return [
            (
                _summary_label(payload["left"]["summary"], suffix="left"),
                payload["left"]["summary"],
            ),
            (
                _summary_label(payload["right"]["summary"], suffix="right"),
                payload["right"]["summary"],
            ),
        ]

    return [(_summary_label(payload, fallback=path.stem), payload)]


def _summary_label(summary, fallback=None, suffix=None):
    requested = summary.get("environment", {}).get("requested_devices") or []
    key = requested[0] if requested else (fallback or "report")
    label = _DISPLAY_NAMES.get(key, key)

    if suffix is not None:
        return f"{label} ({suffix})"

    return label


def _print_summary(label, summary, inventory, show_skip_only):
    env = summary.get("environment", {})
    totals = summary.get("totals", {})
    rows = _build_rows(summary, inventory)
    selected_categories = _selected_categories(summary)

    tested = sum(1 for row in rows if row["status"] != "NO_PYTEST_RECORD")
    pass_gt0 = sum(1 for row in rows if row["passed"] > 0)
    any_skip = sum(1 for row in rows if row["skipped"] > 0)
    skip_only = [row for row in rows if row["status"] == "SKIPPED_ONLY"]
    failed = sum(1 for row in rows if row["status"] == "FAILED")
    no_record = sum(1 for row in rows if row["status"] == "NO_PYTEST_RECORD")

    print(f"{label}")
    print(f"  torch={env.get('torch_version')}")
    if selected_categories:
        print("  operator category filter: " + ", ".join(selected_categories))
    print(
        "  case totals: "
        f"collected={totals.get('collected')} "
        f"passed={totals.get('passed')} "
        f"skipped={totals.get('skipped')} "
        f"failed={totals.get('failed')}"
    )
    print(
        "  operator totals: "
        f"total={len(rows)} "
        f"tested={tested} "
        f"pass>0={pass_gt0} "
        f"any-skip={any_skip} "
        f"skip-only={len(skip_only)} "
        f"failed={failed} "
        f"no-record={no_record}"
    )
    _print_category_summary(rows)

    if show_skip_only and skip_only:
        print("  skip-only operators:")
        for row in skip_only:
            print(
                "    "
                f"{row['operator']} "
                f"(cases={row['cases']}, skipped={row['skipped']}, module={row['module']})"
            )


def _build_rows(summary, inventory):
    selected_categories = set(_selected_categories(summary))
    if selected_categories:
        inventory = [
            item for item in inventory if item.get("category") in selected_categories
        ]

    summary_rows = _aggregate_summary_rows(summary.get("operators", []))
    rows = []

    for item in inventory:
        summary_row = summary_rows.get(item["operator"])

        if summary_row is None:
            rows.append(
                {
                    "operator": item["operator"],
                    "category": item["category"],
                    "status": "NO_PYTEST_RECORD",
                    "cases": 0,
                    "passed": 0,
                    "skipped": 0,
                    "failed": 0,
                    "module": "-",
                }
            )
            continue

        outcomes = summary_row["outcomes"]
        passed = outcomes.get("passed", 0)
        skipped = outcomes.get("skipped", 0)
        failed = outcomes.get("failed", 0)

        if failed > 0:
            status = "FAILED"
        elif passed > 0 and skipped > 0:
            status = "PASSED_WITH_SKIPS"
        elif passed > 0:
            status = "PASSED"
        elif skipped > 0:
            status = "SKIPPED_ONLY"
        else:
            status = "NO_PYTEST_RECORD"

        rows.append(
            {
                "operator": item["operator"],
                "category": item["category"],
                "status": status,
                "cases": summary_row.get("cases", 0),
                "passed": passed,
                "skipped": skipped,
                "failed": failed,
                "module": summary_row.get("module", "-"),
            }
        )

    return rows


def _aggregate_summary_rows(operator_rows):
    grouped = defaultdict(list)

    for row in operator_rows:
        operator = inventory_operator_name(row.get("operator"), row.get("aten_name"))

        if operator is None:
            continue

        grouped[operator].append(row)

    return {
        operator: _aggregate_operator_group(operator, rows)
        for operator, rows in grouped.items()
    }


def _aggregate_operator_group(operator, rows):
    outcomes = Counter()
    skip_reasons = Counter()
    dtypes = set()
    implementation_indices = set()
    modules = set()
    torch_devices = set()

    for row in rows:
        outcomes.update(row.get("outcomes", {}))
        modules.add(row.get("module") or "-")
        torch_devices.add(row.get("torch_device") or "-")
        dtypes.update(dtype for dtype in row.get("dtypes", []) if dtype is not None)
        implementation_indices.update(row.get("implementation_indices", []))

        for entry in row.get("skip_reasons", []):
            skip_reasons[entry.get("reason", "")] += entry.get("count", 0)

    return {
        "operator": operator,
        "cases": sum(row.get("cases", 0) for row in rows),
        "outcomes": {
            "passed": outcomes.get("passed", 0),
            "skipped": outcomes.get("skipped", 0),
            "failed": outcomes.get("failed", 0),
        },
        "dtypes": sorted(dtypes),
        "implementation_indices": sorted(implementation_indices),
        "module": ",".join(sorted(modules)) or "-",
        "torch_device": ",".join(sorted(torch_devices)) or "-",
        "skip_reasons": [
            {"reason": reason, "count": count}
            for reason, count in sorted(
                skip_reasons.items(), key=lambda item: (-item[1], item[0])
            )
        ],
    }


def _selected_categories(summary):
    return tuple(summary.get("filters", {}).get("operator_categories") or ())


def _print_category_summary(rows):
    print("  category totals:")

    for category in CATEGORIES:
        category_rows = [row for row in rows if row["category"] == category]

        if not category_rows:
            continue

        tested = sum(1 for row in category_rows if row["status"] != "NO_PYTEST_RECORD")
        pass_gt0 = sum(1 for row in category_rows if row["passed"] > 0)
        skip_only = sum(1 for row in category_rows if row["status"] == "SKIPPED_ONLY")
        failed = sum(1 for row in category_rows if row["status"] == "FAILED")
        no_record = sum(
            1 for row in category_rows if row["status"] == "NO_PYTEST_RECORD"
        )

        print(
            f"    {category}: "
            f"cases={sum(row['cases'] for row in category_rows)} "
            f"passed={sum(row['passed'] for row in category_rows)} "
            f"skipped={sum(row['skipped'] for row in category_rows)} "
            f"failed={sum(row['failed'] for row in category_rows)} "
            f"operators={len(category_rows)} "
            f"tested={tested} "
            f"pass>0={pass_gt0} "
            f"skip-only={skip_only} "
            f"no-record={no_record} "
            f"failed-ops={failed}"
        )


if __name__ == "__main__":
    main()
