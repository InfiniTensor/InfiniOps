#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_BASE_DIR = _REPO_ROOT / "src" / "base"
_TORCH_OPS_YAML = _REPO_ROOT / "scripts" / "torch_ops.yaml"

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

_STATUS_ORDER = {
    "FAILED": 0,
    "PASSED_WITH_SKIPS": 1,
    "PASSED": 2,
    "SKIPPED_ONLY": 3,
    "NO_PYTEST_RECORD": 4,
}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Render Markdown coverage tables from pytest operator reports "
            "and the source operator inventory."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=pathlib.Path,
        help="Report JSON path(s). Supports single-platform reports and diff reports.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Optional Markdown output path. Defaults to stdout only.",
    )
    args = parser.parse_args()

    inventory = _load_source_inventory()
    platforms = _load_platform_reports(args.inputs)
    markdown = _render_markdown(platforms, inventory)
    print(markdown)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown + "\n", encoding="utf-8")
        print(f"\ncoverage tables written to {args.output}")


def _load_source_inventory():
    inventory = {}

    for path in sorted(_BASE_DIR.glob("*.h")):
        inventory[path.stem] = {
            "operator": path.stem,
            "category": "native",
        }

    for line in _TORCH_OPS_YAML.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()

        if not stripped.startswith("- "):
            continue

        operator = _public_op_name(stripped[2:])
        inventory.setdefault(
            operator,
            {
                "operator": operator,
                "category": "torch-generated",
            },
        )

    return [inventory[name] for name in sorted(inventory)]


def _public_op_name(aten_name):
    public_name = aten_name

    if public_name.startswith("_"):
        public_name = f"aten{public_name}"

    if public_name.endswith("_") and not public_name.endswith("__"):
        public_name = public_name[:-1] + "_inplace"

    return public_name


def _load_platform_reports(paths):
    platforms = []
    seen_labels = Counter()

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))

        for summary in _extract_summaries(payload):
            label = _platform_label(summary)
            seen_labels[label] += 1

            if seen_labels[label] > 1:
                label = f"{label}-{seen_labels[label]}"

            platforms.append(
                {
                    "label": label,
                    "summary": summary,
                }
            )

    return platforms


def _extract_summaries(payload):
    if {"left", "right", "operator_diff", "case_diff"} <= set(payload):
        return [payload["left"]["summary"], payload["right"]["summary"]]

    return [payload]


def _platform_label(summary):
    requested = summary.get("environment", {}).get("requested_devices") or []

    if requested:
        key = requested[0]
    else:
        output_path = summary.get("output_path", "")
        key = pathlib.Path(output_path).stem or "report"

    return _DISPLAY_NAMES.get(key, key)


def _render_markdown(platforms, inventory):
    lines = []
    platform_views = [
        _build_platform_view(platform, inventory)
        for platform in platforms
    ]

    lines.append("# Pytest Operator Coverage")
    lines.append("")
    lines.append(
        f"Source inventory: {len(inventory)} operators "
        f"({sum(1 for item in inventory if item['category'] == 'native')} native, "
        f"{sum(1 for item in inventory if item['category'] == 'torch-generated')} torch-generated)"
    )
    lines.append("")
    lines.extend(_render_platform_summary(platform_views))
    lines.append("")
    lines.extend(_render_category_summary(platform_views))
    lines.append("")
    lines.extend(_render_cross_platform_matrix(platform_views))

    for view in platform_views:
        lines.append("")
        lines.extend(_render_platform_missing(view))
        lines.append("")
        lines.extend(_render_platform_detail(view))

    return "\n".join(lines).rstrip()


def _build_platform_view(platform, inventory):
    summary = platform["summary"]
    summary_rows = {row["operator"]: row for row in summary.get("operators", [])}
    detail_rows = []

    for item in inventory:
        summary_row = summary_rows.get(item["operator"])
        detail_rows.append(_build_detail_row(item, summary_row))

    return {
        "label": platform["label"],
        "summary": summary,
        "rows": detail_rows,
    }


def _build_detail_row(item, summary_row):
    if summary_row is None:
        return {
            "operator": item["operator"],
            "category": item["category"],
            "status": "NO_PYTEST_RECORD",
            "cases": 0,
            "passed": 0,
            "skipped": 0,
            "failed": 0,
            "module": "-",
            "torch_device": "-",
            "implementation_indices": [],
            "dtypes": [],
            "skip_reasons": [],
        }

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

    return {
        "operator": item["operator"],
        "category": item["category"],
        "status": status,
        "cases": summary_row.get("cases", 0),
        "passed": passed,
        "skipped": skipped,
        "failed": failed,
        "module": summary_row.get("module", "-"),
        "torch_device": summary_row.get("torch_device", "-"),
        "implementation_indices": summary_row.get("implementation_indices", []),
        "dtypes": summary_row.get("dtypes", []),
        "skip_reasons": summary_row.get("skip_reasons", []),
    }


def _render_platform_summary(platform_views):
    lines = []

    lines.append("## Platform Summary")
    lines.append("")
    lines.append(
        "| Platform | Total Ops | Tested Ops | Pass>0 Ops | Skip-only Ops | Failed Ops | No Pytest Record |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")

    for view in platform_views:
        rows = view["rows"]
        lines.append(
            "| "
            f"{view['label']} | "
            f"{len(rows)} | "
            f"{sum(row['status'] != 'NO_PYTEST_RECORD' for row in rows)} | "
            f"{sum(row['passed'] > 0 for row in rows)} | "
            f"{sum(row['status'] == 'SKIPPED_ONLY' for row in rows)} | "
            f"{sum(row['failed'] > 0 for row in rows)} | "
            f"{sum(row['status'] == 'NO_PYTEST_RECORD' for row in rows)} |"
        )

    return lines


def _render_category_summary(platform_views):
    lines = []

    lines.append("## Category Summary")
    lines.append("")
    lines.append(
        "| Platform | Native Total | Native Tested | Native Pass>0 | Native Skip-only | Native No Record | Generated Total | Generated Tested | Generated Pass>0 | Generated Skip-only | Generated No Record |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    for view in platform_views:
        native_rows = [row for row in view["rows"] if row["category"] == "native"]
        generated_rows = [
            row for row in view["rows"] if row["category"] == "torch-generated"
        ]
        lines.append(
            "| "
            f"{view['label']} | "
            f"{len(native_rows)} | "
            f"{sum(row['status'] != 'NO_PYTEST_RECORD' for row in native_rows)} | "
            f"{sum(row['passed'] > 0 for row in native_rows)} | "
            f"{sum(row['status'] == 'SKIPPED_ONLY' for row in native_rows)} | "
            f"{sum(row['status'] == 'NO_PYTEST_RECORD' for row in native_rows)} | "
            f"{len(generated_rows)} | "
            f"{sum(row['status'] != 'NO_PYTEST_RECORD' for row in generated_rows)} | "
            f"{sum(row['passed'] > 0 for row in generated_rows)} | "
            f"{sum(row['status'] == 'SKIPPED_ONLY' for row in generated_rows)} | "
            f"{sum(row['status'] == 'NO_PYTEST_RECORD' for row in generated_rows)} |"
        )

    return lines


def _render_cross_platform_matrix(platform_views):
    lines = []
    labels = [view["label"] for view in platform_views]
    row_maps = {
        view["label"]: {row["operator"]: row for row in view["rows"]}
        for view in platform_views
    }
    operators = [row["operator"] for row in platform_views[0]["rows"]]
    categories = {row["operator"]: row["category"] for row in platform_views[0]["rows"]}

    lines.append("## Cross-Platform Matrix")
    lines.append("")
    lines.append(
        "| Operator | Category | "
        + " | ".join(labels)
        + " |"
    )
    lines.append(
        "| --- | --- | "
        + " | ".join("---" for _ in labels)
        + " |"
    )

    for operator in operators:
        cells = [
            _matrix_cell(row_maps[label][operator])
            for label in labels
        ]
        lines.append(
            "| "
            f"{operator} | {categories[operator]} | "
            + " | ".join(cells)
            + " |"
        )

    return lines


def _matrix_cell(row):
    if row["status"] == "NO_PYTEST_RECORD":
        return "NO_PYTEST_RECORD"

    return (
        f"{row['status']} "
        f"(P={row['passed']}, S={row['skipped']}, F={row['failed']})"
    )


def _render_platform_missing(view):
    missing = [row["operator"] for row in view["rows"] if row["status"] == "NO_PYTEST_RECORD"]

    lines = []
    lines.append(f"## {view['label']} Missing From Pytest")
    lines.append("")
    lines.append(f"- Count: {len(missing)}")
    lines.append(f"- Operators: {', '.join(missing) if missing else '-'}")

    return lines


def _render_platform_detail(view):
    lines = []

    lines.append(f"## {view['label']} Detailed Coverage")
    lines.append("")
    lines.append(
        "| Operator | Category | Status | Cases | Passed | Skipped | Failed | Test Module | Device | Impl | Top Skip Reason |"
    )
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |")

    sorted_rows = sorted(
        view["rows"],
        key=lambda row: (
            row["category"] != "native",
            _STATUS_ORDER[row["status"]],
            row["operator"],
        ),
    )

    for row in sorted_rows:
        skip_reason = row["skip_reasons"][0]["reason"] if row["skip_reasons"] else "-"
        impls = ",".join(str(value) for value in row["implementation_indices"]) or "-"
        lines.append(
            "| "
            f"{row['operator']} | "
            f"{row['category']} | "
            f"{row['status']} | "
            f"{row['cases']} | "
            f"{row['passed']} | "
            f"{row['skipped']} | "
            f"{row['failed']} | "
            f"{row['module']} | "
            f"{row['torch_device']} | "
            f"{impls} | "
            f"{skip_reason.replace('|', '/')} |"
        )

    return lines


if __name__ == "__main__":
    main()
