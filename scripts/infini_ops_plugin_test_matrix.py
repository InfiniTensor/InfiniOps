#!/usr/bin/env python3
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import infini_ops_plugin_registry

CORE_PATHS = {
    "CMakeLists.txt",
    ".github/ci_config.yml",
    "docs/plugin_contract.md",
    "scripts/generate_wrappers.py",
    "scripts/infini_ops_plugin_registry.py",
    "scripts/infini_ops_plugin_test_matrix.py",
    "src/CMakeLists.txt",
}

CORE_PREFIXES = (
    ".github/workflows/",
    "cmake/",
    "include/",
)


def _normalize_path(path):
    return pathlib.PurePosixPath(str(path).replace("\\", "/")).as_posix()


def _matches(path, root):
    return path == root or path.startswith(f"{root}/")


def _device_plugins(manifests):
    return [name for name, manifest in manifests.items() if manifest["kind"] == "device"]


def _dependent_plugins(manifests, plugin_name):
    return [
        name
        for name, manifest in manifests.items()
        if plugin_name in manifest["depends"]
    ]


def _expand_plugins(manifests, plugin_names):
    expanded = []

    def add(name):
        if name not in expanded:
            expanded.append(name)

    for name in plugin_names:
        add(name)
        if manifests[name]["kind"] == "shared":
            for dependent in _dependent_plugins(manifests, name):
                add(dependent)

    return expanded


def _manifest_roots(manifests):
    roots = []
    for name, manifest in manifests.items():
        roots.append((f"plugins/{name}", name))
        for field in ("source_roots", "operator_roots"):
            for root in manifest[field]:
                roots.append((_normalize_path(root), name))
        for header in manifest["device_headers"].values():
            header = _normalize_path(header)
            roots.append((header, name))
            roots.append((f"src/{header}", name))

    return roots


def _plugins_for_path(manifests, path):
    path = _normalize_path(path)

    if path in CORE_PATHS or any(path.startswith(prefix) for prefix in CORE_PREFIXES):
        return _device_plugins(manifests), True

    matches = [
        (len(root), name)
        for root, name in _manifest_roots(manifests)
        if _matches(path, root)
    ]

    if not matches:
        return _device_plugins(manifests), True

    longest = max(length for length, _ in matches)
    plugin_names = [name for length, name in matches if length == longest]

    return _expand_plugins(manifests, plugin_names), False


def _append_unique(values, new_values):
    for value in new_values:
        if value not in values:
            values.append(value)


def build_test_matrix(plugin_root, changed_paths):
    manifests = infini_ops_plugin_registry.load_plugin_manifests(plugin_root)
    plugins = []
    requires_full_matrix = False

    for path in changed_paths:
        path_plugins, path_requires_full_matrix = _plugins_for_path(manifests, path)
        _append_unique(plugins, path_plugins)
        requires_full_matrix = requires_full_matrix or path_requires_full_matrix

    devices = []
    test_devices = []
    for plugin in plugins:
        manifest = manifests[plugin]
        _append_unique(devices, manifest["devices"])
        _append_unique(test_devices, manifest["test_devices"].values())

    ci_platforms = [device for device in devices if device != "cpu"]

    return {
        "plugins": plugins,
        "devices": devices,
        "test_devices": test_devices,
        "ci_platforms": ci_platforms,
        "requires_full_matrix": requires_full_matrix,
    }


def _read_paths(args):
    if args.paths:
        return args.paths

    return [line.strip() for line in sys.stdin if line.strip()]


def _print_github_output(matrix):
    platform = ",".join(matrix["ci_platforms"])
    requires_full_matrix = str(matrix["requires_full_matrix"]).lower()
    compact_json = json.dumps(matrix, sort_keys=True, separators=(",", ":"))

    print(f"platform={platform}")
    print(f"requires_full_matrix={requires_full_matrix}")
    print(f"matrix_json={compact_json}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Map changed paths to `infini::ops` plugin test devices."
    )
    parser.add_argument(
        "--plugin-root",
        action="append",
        default=None,
        help=(
            "Directory containing `plugin.json` manifests. Pass multiple times "
            "to include external plugin roots."
        ),
    )
    parser.add_argument(
        "--github-output",
        action="store_true",
        help="Print `$GITHUB_OUTPUT` compatible key/value lines.",
    )
    parser.add_argument("paths", nargs="*", help="Changed paths to classify.")
    args = parser.parse_args(argv)

    matrix = build_test_matrix(args.plugin_root or ["plugins"], _read_paths(args))
    if args.github_output:
        _print_github_output(matrix)
    else:
        print(json.dumps(matrix, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
