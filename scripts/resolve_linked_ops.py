import argparse
import dataclasses
import fnmatch
import importlib.metadata
import json
import os
import pathlib
import re
import subprocess
import sys

import yaml


_PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
_DEFAULT_SOURCE_ROOT = _PROJECT_DIR / "src" / "linked"
_DEFAULT_OUTPUT_DIR = _PROJECT_DIR / "generated" / "linked"
_LIBRARY_KEYS = {
    "transport",
    "python_distribution_package",
    "library_glob",
}
_BINDING_KEYS = {"library", "required_symbols"}
_SUPPORTED_TRANSPORTS = {"torch"}


class ResolutionError(RuntimeError):
    pass


class _StrictLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)

    return mapping


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


@dataclasses.dataclass(frozen=True)
class LibraryConfig:
    device: str
    family: str
    name: str
    path: pathlib.Path
    transport: str
    python_distribution_package: str
    library_glob: str


@dataclasses.dataclass(frozen=True)
class BindingConfig:
    device: str
    family: str
    implementation: str
    name: str
    path: pathlib.Path
    source: pathlib.Path
    library: str
    required_symbols: tuple[str, ...]


def _load_yaml_mapping(path, expected_keys):
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_StrictLoader)
    except (OSError, yaml.YAMLError) as error:
        raise ResolutionError(f"failed to read {path}: {error}") from error

    if not isinstance(data, dict):
        raise ResolutionError(f"{path} must contain a YAML mapping")

    keys = set(data)
    unknown_keys = sorted(keys - expected_keys)
    missing_keys = sorted(expected_keys - keys)
    if unknown_keys:
        raise ResolutionError(
            f"{path} contains unknown keys: {', '.join(unknown_keys)}"
        )
    if missing_keys:
        raise ResolutionError(
            f"{path} is missing required keys: {', '.join(missing_keys)}"
        )

    return data


def _require_string(data, key, path):
    value = data[key]
    if not isinstance(value, str) or not value.strip():
        raise ResolutionError(f"{path}: {key} must be a non-empty string")
    return value.strip()


def _require_relative_glob(data, key, path):
    value = _require_string(data, key, path)
    normalized = pathlib.PurePosixPath(value.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ResolutionError(f"{path}: {key} must be relative to the distribution")
    return normalized.as_posix()


def _find_platform_dir(source_root, device):
    candidates = []
    direct = source_root / device
    if direct.is_dir():
        candidates.append(direct)
    candidates.extend(
        path
        for path in sorted(source_root.glob(f"*/{device}"))
        if path.is_dir() and path != direct
    )

    if len(candidates) > 1:
        formatted = ", ".join(str(path) for path in candidates)
        raise ResolutionError(
            f"multiple linked platform directories found for {device}: {formatted}"
        )

    return candidates[0] if candidates else None


def _platform_family(source_root, platform_dir):
    relative = platform_dir.relative_to(source_root)
    return relative.parts[-2] if len(relative.parts) > 1 else "generic"


def _load_libraries(source_root, platform_dir, device):
    family = _platform_family(source_root, platform_dir)
    libraries = {}
    for path in sorted(platform_dir.glob("*.yaml")):
        data = _load_yaml_mapping(path, _LIBRARY_KEYS)
        transport = _require_string(data, "transport", path)
        if transport not in _SUPPORTED_TRANSPORTS:
            supported = ", ".join(sorted(_SUPPORTED_TRANSPORTS))
            raise ResolutionError(
                f"{path}: unsupported transport {transport!r}; expected {supported}"
            )

        libraries[path.stem] = LibraryConfig(
            device=device,
            family=family,
            name=path.stem,
            path=path,
            transport=transport,
            python_distribution_package=_require_string(
                data, "python_distribution_package", path
            ),
            library_glob=_require_relative_glob(data, "library_glob", path),
        )

    return libraries


def _load_bindings(source_root, platform_dir, device, selected_ops):
    family = _platform_family(source_root, platform_dir)
    bindings = []
    for path in sorted((platform_dir / "ops").glob("*/*.yaml")):
        name = path.parent.name
        if selected_ops is not None and name not in selected_ops:
            continue

        data = _load_yaml_mapping(path, _BINDING_KEYS)
        symbols = data["required_symbols"]
        if (
            not isinstance(symbols, list)
            or not symbols
            or any(
                not isinstance(symbol, str) or not symbol.strip() for symbol in symbols
            )
        ):
            raise ResolutionError(
                f"{path}: required_symbols must be a non-empty list of strings"
            )
        symbols = tuple(symbol.strip() for symbol in symbols)
        if len(symbols) != len(set(symbols)):
            raise ResolutionError(f"{path}: required_symbols contains duplicates")

        header = path.with_suffix(".h")
        source = path.with_suffix(".cc")
        if not header.is_file():
            raise ResolutionError(f"{path}: missing sibling {header.name}")
        if not source.is_file():
            raise ResolutionError(f"{path}: missing sibling {source.name}")

        bindings.append(
            BindingConfig(
                device=device,
                family=family,
                implementation=path.stem,
                name=name,
                path=path,
                source=source.resolve(),
                library=_require_string(data, "library", path),
                required_symbols=symbols,
            )
        )

    return bindings


def _matches_library_glob(relative, pattern):
    relative_parts = pathlib.PurePosixPath(relative).parts
    pattern_parts = pathlib.PurePosixPath(pattern).parts
    return len(relative_parts) == len(pattern_parts) and all(
        fnmatch.fnmatchcase(part, pattern_part)
        for part, pattern_part in zip(relative_parts, pattern_parts)
    )


def _locate_distribution_library(config):
    try:
        distribution = importlib.metadata.distribution(
            config.python_distribution_package
        )
    except importlib.metadata.PackageNotFoundError as error:
        raise ResolutionError(
            f"Python distribution package "
            f"{config.python_distribution_package!r} required by "
            f"{config.path} is not installed"
        ) from error

    matches = []
    distribution_root = pathlib.Path(distribution.locate_file("")).resolve()
    for entry in distribution.files or ():
        relative = pathlib.PurePosixPath(str(entry).replace("\\", "/"))
        if (
            not relative.is_absolute()
            and ".." not in relative.parts
            and _matches_library_glob(relative, config.library_glob)
        ):
            candidate = pathlib.Path(distribution.locate_file(entry)).resolve()
            if candidate.is_file() and candidate.is_relative_to(distribution_root):
                matches.append(candidate)

    # Some vendor wheels intentionally ship sparse or empty RECORD metadata.
    # Keep the YAML contract unchanged and fall back to the distribution root.
    for candidate in distribution_root.glob(config.library_glob):
        candidate = candidate.resolve()
        if candidate.is_file() and candidate.is_relative_to(distribution_root):
            matches.append(candidate)

    matches = sorted(set(matches))
    if len(matches) != 1:
        formatted = ", ".join(str(path) for path in matches) or "none"
        raise ResolutionError(
            f"{config.path}: library_glob {config.library_glob!r} matched "
            f"{len(matches)} files in "
            f"{config.python_distribution_package!r}: {formatted}"
        )

    return matches[0]


def _run_symbol_tool(command, library_path):
    try:
        result = subprocess.run(
            [*command, str(library_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        stderr = getattr(error, "stderr", "") or ""
        detail = f": {stderr.strip()}" if stderr.strip() else ""
        raise ResolutionError(
            f"failed to inspect {library_path} with {command[0]}{detail}"
        ) from error

    return result.stdout


def _parse_nm_symbols(output):
    symbols = set()
    for line in output.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) == 3:
            symbols.add(fields[2])
    return symbols


def _parse_readelf_symbols(output):
    symbols = set()
    for line in output.splitlines():
        fields = line.strip().split(maxsplit=7)
        if (
            len(fields) == 8
            and fields[0].endswith(":")
            and fields[3] == "FUNC"
            and fields[4] == "GLOBAL"
            and fields[5] == "DEFAULT"
            and fields[6] != "UND"
        ):
            symbols.add(fields[7])
    return symbols


def _inspect_dynamic_symbols(library_path, nm, readelf):
    nm_output = _run_symbol_tool([nm, "-D", "--defined-only", "-C"], library_path)
    readelf_output = _run_symbol_tool(
        [readelf, "--dyn-syms", "--wide", "--demangle"], library_path
    )
    return _parse_nm_symbols(nm_output), _parse_readelf_symbols(readelf_output)


def _matches_required_symbol(exported, required):
    return exported == required


def _verify_required_symbols(config, library_path, nm_symbols, readelf_symbols):
    for required in config.required_symbols:
        missing_from = []
        if not any(
            _matches_required_symbol(exported, required) for exported in nm_symbols
        ):
            missing_from.append("nm")
        if not any(
            _matches_required_symbol(exported, required) for exported in readelf_symbols
        ):
            missing_from.append("readelf")
        if missing_from:
            raise ResolutionError(
                f"{config.path}: required symbol {required!r} is not exported by "
                f"{library_path} according to {' and '.join(missing_from)}"
            )


def _cmake_quote(value):
    return str(value).replace("\\", "/").replace(";", "\\;").replace('"', '\\"')


def _render_cmake_manifest(payload):
    variables = {
        "INFINI_OPS_LINKED_SOURCES": [
            operator["source"] for operator in payload["operators"]
        ],
        "INFINI_OPS_LINKED_LIBRARIES": [
            library["path"] for library in payload["libraries"]
        ],
        "INFINI_OPS_LINKED_RUNTIME_DIRS": [
            library["runtime_dir"] for library in payload["libraries"]
        ],
        "INFINI_OPS_LINKED_TRANSPORTS": [
            library["transport"] for library in payload["libraries"]
        ],
    }

    lines = ["# Generated by scripts/resolve_linked_ops.py. Do not edit."]
    for name, values in variables.items():
        lines.append(f"set({name}")
        lines.extend(f'    "{_cmake_quote(value)}"' for value in sorted(set(values)))
        lines.append(")")
        lines.append("")

    return "\n".join(lines)


def _write_if_changed(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def _normalize_values(values):
    if values is None:
        return None
    return tuple(
        dict.fromkeys(
            item.strip().lower()
            for value in values
            for item in re.split(r"[;,]", value)
            if item.strip()
        )
    )


def resolve_linked_ops(
    devices,
    ops=None,
    *,
    source_root=_DEFAULT_SOURCE_ROOT,
    output_dir=_DEFAULT_OUTPUT_DIR,
    nm="nm",
    readelf="readelf",
):
    source_root = pathlib.Path(source_root).resolve()
    output_dir = pathlib.Path(output_dir).resolve()
    devices = _normalize_values(devices) or ()
    selected_ops = _normalize_values(ops)
    selected_op_set = set(selected_ops) if selected_ops is not None else None

    bindings = []
    library_configs = {}
    for device in devices:
        platform_dir = _find_platform_dir(source_root, device)
        if platform_dir is None:
            continue

        libraries = _load_libraries(source_root, platform_dir, device)
        for name, config in libraries.items():
            library_configs[(device, name)] = config
        bindings.extend(
            _load_bindings(source_root, platform_dir, device, selected_op_set)
        )

    bindings.sort(
        key=lambda binding: (
            binding.device,
            binding.name,
            binding.implementation,
        )
    )
    resolved_libraries = {}
    inspected_symbols = {}
    for binding in bindings:
        key = (binding.device, binding.library)
        library_config = library_configs.get(key)
        if library_config is None:
            raise ResolutionError(
                f"{binding.path}: unknown library {binding.library!r} for "
                f"device {binding.device}"
            )

        if key not in resolved_libraries:
            library_path = _locate_distribution_library(library_config)
            resolved_libraries[key] = library_path
            inspected_symbols[key] = _inspect_dynamic_symbols(library_path, nm, readelf)

        library_path = resolved_libraries[key]
        nm_symbols, readelf_symbols = inspected_symbols[key]
        _verify_required_symbols(binding, library_path, nm_symbols, readelf_symbols)

    required_symbols = {
        symbol for binding in bindings for symbol in binding.required_symbols
    }
    for required_symbol in sorted(required_symbols):
        exporters = {
            resolved_libraries[key]
            for key, (nm_symbols, readelf_symbols) in inspected_symbols.items()
            if required_symbol in nm_symbols and required_symbol in readelf_symbols
        }
        if len(exporters) > 1:
            formatted = ", ".join(str(path) for path in sorted(exporters))
            raise ResolutionError(
                f"required symbol {required_symbol!r} is exported by multiple "
                f"linked libraries: {formatted}"
            )

    libraries_by_basename = {}
    for library_path in resolved_libraries.values():
        previous = libraries_by_basename.setdefault(library_path.name, library_path)
        if previous != library_path:
            raise ResolutionError(
                f"linked libraries share basename {library_path.name!r}: "
                f"{previous} and {library_path}"
            )

    libraries = []
    for key in sorted(resolved_libraries):
        config = library_configs[key]
        library_path = resolved_libraries[key]
        libraries.append(
            {
                "device": config.device,
                "family": config.family,
                "name": config.name,
                "path": str(library_path),
                "python_distribution_package": (config.python_distribution_package),
                "runtime_dir": str(library_path.parent),
                "transport": config.transport,
            }
        )

    payload = {
        "devices": list(devices),
        "libraries": libraries,
        "operators": [
            {
                "device": binding.device,
                "family": binding.family,
                "library": binding.library,
                "implementation": binding.implementation,
                "name": binding.name,
                "required_symbols": list(binding.required_symbols),
                "source": str(binding.source),
            }
            for binding in bindings
        ],
    }
    _write_if_changed(output_dir / "manifest.cmake", _render_cmake_manifest(payload))
    _write_if_changed(
        output_dir / "resolved.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )

    return payload


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Resolve installed linked operator libraries for InfiniOps."
    )
    parser.add_argument("--devices", nargs="+", required=True)
    parser.add_argument("--ops", nargs="*")
    parser.add_argument("--source-root", default=_DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--nm", default=os.environ.get("CMAKE_NM", "nm"))
    parser.add_argument("--readelf", default=os.environ.get("CMAKE_READELF", "readelf"))
    return parser.parse_args()


def main():
    args = _parse_args()
    ops = args.ops
    if ops is None and "INFINI_OPS_OPS" in os.environ:
        ops = [os.environ["INFINI_OPS_OPS"]]
    try:
        resolve_linked_ops(
            args.devices,
            ops,
            source_root=args.source_root,
            output_dir=args.output_dir,
            nm=args.nm,
            readelf=args.readelf,
        )
    except ResolutionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
