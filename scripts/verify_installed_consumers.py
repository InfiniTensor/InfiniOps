#!/usr/bin/env python3
"""Verify a Linux CPU Add/Gemm installation without changing legacy test behavior."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "test_cpp_operator_call_instantiation_smoke",
    "test_cpp_operator_call_trace_is_json",
    "test_cpp_returning_call_smoke",
)


def validate_junit(path, required=REQUIRED):
    root = ET.parse(path).getroot()
    if root.tag not in ("testsuite", "testsuites"):
        raise ValueError("Invalid JUnit document root.")
    cases = list(root.iter("testcase"))
    names = [case.get("name") for case in cases]

    if not required or len(names) != len(required) or set(names) != set(required):
        raise ValueError("Required consumer tests are absent, duplicated, or empty.")

    for case in cases:
        if any(case.find(tag) is not None for tag in ("skipped", "failure", "error")):
            raise ValueError(f"Consumer test did not pass: `{case.get('name')}`.")

    for node in root.iter():
        if node.tag in ("testsuite", "testsuites"):
            if any(node.find(tag) is not None for tag in ("error", "failure")):
                raise ValueError("JUnit contains a suite-level failure.")
            if "tests" in node.attrib and node.get("tests") != str(
                len(list(node.iter("testcase")))
            ):
                raise ValueError("JUnit case count is inconsistent.")
            for key in ("errors", "failures", "skipped"):
                if node.get(key, "0") != "0":
                    raise ValueError(f"JUnit reports `{key}`.")

    return names


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(prefix, rt_prefix, output):
    prefix, rt_prefix, output = (p.resolve() for p in (prefix, rt_prefix, output))

    if output.is_relative_to(ROOT) or ROOT.is_relative_to(output):
        raise ValueError("Consumer output must be outside the source tree.")
    if any(p.is_relative_to(ROOT) for p in (prefix, rt_prefix)):
        raise ValueError("Consumer installations must be outside the source tree.")

    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "failed", "scope": "linux-cpu-add-gemm", "commands": []}

    def run(command, name, *, env=None):
        completed = subprocess.run(
            command,
            cwd=output,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (output / f"{name}.log").write_text(completed.stdout, encoding="utf-8")
        report["commands"].append(
            {
                "argv": list(map(str, command)),
                "returncode": completed.returncode,
                "log": f"{name}.log",
            }
        )

        if completed.returncode:
            raise RuntimeError(f"Consumer command `{name}` failed; see its log.")

        return completed.stdout

    try:
        if platform.system() != "Linux" or platform.machine() not in (
            "x86_64",
            "AMD64",
        ):
            raise ValueError("This recipe requires Linux x86_64.")

        for path in (
            prefix / "include/infini/ops.h",
            rt_prefix / "include/infini/rt.h",
        ):
            if not path.is_file():
                raise ValueError(f"Missing installed header `{path}`.")

        for tool in ("cmake", os.environ.get("CXX", "c++"), "nm", "readelf"):
            if not shutil.which(tool):
                raise ValueError(f"Required tool `{tool}` is unavailable.")

        env = os.environ.copy()
        for key in ("PYTHONPATH", "PYTHONHOME", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
            env.pop(key, None)
        env.update(
            {
                "INFINI_OPS_INSTALL_PREFIX": str(prefix),
                "INFINI_RT_ROOT": str(rt_prefix),
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            }
        )
        report["python"] = sys.version
        report["platform"] = platform.platform()
        report["core_revision"] = run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], "core-revision", env=env
        ).strip()
        report["rt_revision"] = env.get("INFINI_RT_REVISION")
        report["verification_inputs"] = {
            name: sha256(ROOT / name)
            for name in (
                "tests/test_cpp_api.py",
                "verification/cpu/consumer.py",
                "verification/cpu/consumer.cc",
                "verification/cpu/CMakeLists.txt",
                "verification/cpu/requirements.lock",
            )
        }
        report["dependencies"] = run(
            [sys.executable, "-I", "-m", "pip", "freeze", "--all"],
            "dependencies",
            env=env,
        )
        run(["cmake", "--version"], "cmake-version", env=env)
        run([os.environ.get("CXX", "c++"), "--version"], "compiler-version", env=env)

        # Copy only existing standalone consumers, not the device conftest or sources.
        shutil.copyfile(ROOT / "tests/test_cpp_api.py", output / "test_cpp_api.py")
        (output / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        run(
            [
                sys.executable,
                "-I",
                "-m",
                "pytest",
                "-c",
                str(output / "pytest.ini"),
                *(str(output / "test_cpp_api.py") + "::" + name for name in REQUIRED),
                "--junitxml=" + str(output / "consumers.xml"),
                "-q",
            ],
            "cpp-tests",
            env=env,
        )
        report["required_cases"] = validate_junit(output / "consumers.xml")

        consumer = output / "consumer-source"
        shutil.copytree(
            ROOT / "verification/cpu",
            consumer,
            ignore=shutil.ignore_patterns("requirements.*"),
        )
        run(
            [
                "cmake",
                "-S",
                str(consumer),
                "-B",
                str(output / "consumer-build"),
                "-DCMAKE_PREFIX_PATH=" + str(prefix) + ";" + str(rt_prefix),
                "-DCMAKE_CXX_COMPILER=" + os.environ.get("CXX", "c++"),
            ],
            "consumer-configure",
            env=env,
        )
        run(
            ["cmake", "--build", str(output / "consumer-build"), "--parallel", "2"],
            "consumer-build",
            env=env,
        )
        selection = json.loads(
            run([str(output / "consumer-build/consumer")], "selection", env=env)
        )
        if selection != {op: {"default": 0, "active": [0]} for op in ("add", "gemm")}:
            raise ValueError("Installed CPU selection differs from the recipe.")
        report["selection"] = selection
        report["python_consumer"] = json.loads(
            run(
                [sys.executable, "-I", str(consumer / "consumer.py"), str(prefix)],
                "python-consumer",
                env=env,
            )
        )

        libraries = sorted(prefix.rglob("libinfiniops.so"))
        if not libraries or len({sha256(path) for path in libraries}) != 1:
            raise ValueError(
                "Missing or mismatched installed InfiniOps library copies."
            )
        report["artifacts"] = {
            str(path.relative_to(prefix)): sha256(path)
            for path in sorted(prefix.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts
        }
        run(["nm", "-D", "--defined-only", str(libraries[0])], "symbols", env=env)
        run(["readelf", "-d", str(libraries[0])], "dynamic-dependencies", env=env)
        report["status"] = "passed"
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        (output / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--rt-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        verify(args.prefix, args.rt_prefix, args.output)
        return 0
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
