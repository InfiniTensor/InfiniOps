#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/skip_stats.sh [cambricon|metax|cpu|PATH-TO-REPORT] [--show-skip-only]

Examples:
  scripts/skip_stats.sh cambricon
  scripts/skip_stats.sh metax
  scripts/skip_stats.sh reports/cambricon.json --show-skip-only
EOF
}

if (($# == 0)); then
    usage >&2
    exit 1
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
summary_script="${repo_root}/scripts/summarize_op_report.py"
first_arg="$1"
shift

case "$first_arg" in
    cambricon|metax|cpu)
        report_path="${repo_root}/reports/${first_arg}.json"
        ;;
    *)
        report_path="$first_arg"
        ;;
esac

if [[ "$report_path" != /* ]]; then
    report_path="${repo_root}/${report_path}"
fi

if [[ ! -f "$report_path" ]]; then
    echo "Report not found: $report_path" >&2
    exit 1
fi

python3 "$summary_script" "$report_path" "$@"
