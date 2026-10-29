"""Stdlib-only P0 tests; can run without importing the device pytest fixtures."""

import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline = load("baseline", "scripts/compatibility_baseline.py")


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for path in (
            "include/infini/ops.h",
            "src/base/add.h",
            "src/operator.h",
            "src/config.h",
            "CMakeLists.txt",
            ".github/ci_config.yml",
        ):
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("baseline\n", encoding="utf-8")
        self.expected = {
            "format_version": 1,
            "source_revision": "a" * 40,
            "scope": baseline.GROUPS,
            "groups": baseline.inventory(self.root),
        }

    def test_unchanged_and_line_endings(self):
        (self.root / "src/base/add.h").write_bytes(b"baseline\r\n")
        self.assertEqual(
            baseline.compare(self.expected, baseline.inventory(self.root)), {}
        )

    def test_signature_default_and_build_policy_changes_require_review(self):
        for path in ("src/base/add.h", "src/config.h", ".github/ci_config.yml"):
            target = self.root / path
            target.write_text("changed\n", encoding="utf-8")
        changes = baseline.compare(self.expected, baseline.inventory(self.root))
        self.assertEqual(len(changes), 2)
        self.assertEqual(
            changes["declaration_inputs"]["changed"], ["src/base/add.h", "src/config.h"]
        )

    def test_added_and_removed_declarations(self):
        target = self.root / "src/base/extra.h"
        target.write_text("new", encoding="utf-8")
        self.assertEqual(
            baseline.compare(self.expected, baseline.inventory(self.root))[
                "declaration_inputs"
            ]["added"],
            ["src/base/extra.h"],
        )
        old = baseline.inventory(self.root)
        target.unlink()
        previous = {**self.expected, "groups": old}
        self.assertEqual(
            baseline.compare(previous, baseline.inventory(self.root))[
                "declaration_inputs"
            ]["removed"],
            ["src/base/extra.h"],
        )

    def test_partial_tree_and_empty_baseline_fail(self):
        invalid = copy.deepcopy(self.expected)
        invalid["groups"]["declaration_inputs"] = {}
        with self.assertRaises(ValueError):
            baseline.compare(invalid, baseline.inventory(self.root))
        (self.root / "src/operator.h").unlink()
        with self.assertRaises(ValueError):
            baseline.inventory(self.root)

    def test_unknown_format_or_reduced_scope_rejected(self):
        for invalid in (
            [],
            {**self.expected, "format_version": 2},
            {**self.expected, "format_version": True},
            {**self.expected, "scope": {}},
            {**self.expected, "groups": {}},
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    baseline.compare(invalid, baseline.inventory(self.root))

    def test_invalid_revision_path_and_digest_rejected(self):
        for revision in (None, "master", "A" * 40):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                baseline.compare(
                    {**self.expected, "source_revision": revision},
                    baseline.inventory(self.root),
                )

        for path, digest in (("../escape.h", "a" * 64), ("src/config.h", "invalid")):
            invalid = copy.deepcopy(self.expected)
            invalid["groups"]["declaration_inputs"][path] = digest

            with self.subTest(path=path), self.assertRaises(ValueError):
                baseline.compare(invalid, baseline.inventory(self.root))

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaises(ValueError):
            baseline.strict_object([("format_version", 1), ("format_version", 1)])

    def test_missing_and_malformed_baseline_cli_fail(self):
        path = self.root / "snapshot.json"
        command = [
            sys.executable,
            str(ROOT / "scripts/compatibility_baseline.py"),
            "check",
            "--root",
            str(self.root),
            "--baseline",
            str(path),
        ]

        for contents in (None, "not json", '{"format_version":1,"format_version":1}'):
            if contents is not None:
                path.write_text(contents, encoding="utf-8")

            with self.subTest(contents=contents):
                result = subprocess.run(command, capture_output=True)
                self.assertEqual(result.returncode, 2)

    def test_implementation_body_outside_scope_does_not_trigger_review(self):
        path = self.root / "src/native/cpu/ops/add/add.cc"
        path.parent.mkdir(parents=True)
        path.write_text("implementation", encoding="utf-8")
        self.assertEqual(
            baseline.compare(self.expected, baseline.inventory(self.root)), {}
        )

    def test_core_types_traits_and_install_template_changes_require_review(self):
        paths = (
            "src/device.h",
            "src/data_type.h",
            "src/common/traits.h",
            "cmake/InfiniOpsConfig.cmake.in",
        )
        for name in paths:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("original", encoding="utf-8")
        previous = {**self.expected, "groups": baseline.inventory(self.root)}
        for name in paths:
            (self.root / name).write_text("changed", encoding="utf-8")
        delta = baseline.compare(previous, baseline.inventory(self.root))
        self.assertEqual(delta["declaration_inputs"]["changed"], sorted(paths[:3]))
        self.assertEqual(delta["build_and_dependency_recipes"]["changed"], [paths[3]])

    def test_selection_configuration_add_change_and_remove(self):
        self.check_recipe_lifecycle(
            "dispatch_and_generation",
            ("ops.json", "scripts/configs/selection.json"),
        )

    def test_nested_build_dependency_and_compiler_inputs(self):
        self.check_recipe_lifecycle(
            "build_and_dependency_recipes",
            (
                "scripts/mcc_wrapper.sh",
                "scripts/mxcc_wrapper.sh",
                "scripts/htcc_wrapper.sh",
                "scripts/dev/build.sh",
                "scripts/prepare_marlin.py",
                "src/native/ascend/custom/CMakeLists.txt",
                "src/native/ascend/custom/build.sh",
                "src/native/ascend/custom/cmake/detect_soc.cmake",
                "src/linked/torch/nvidia/vllm.yaml",
                ".github/workflows/ci_test.yml",
            ),
        )

    def test_installed_consumer_recipe_inputs(self):
        self.check_recipe_lifecycle(
            "build_and_dependency_recipes",
            (
                "verification/cpu/requirements.in",
                "verification/cpu/requirements.lock",
                "verification/cpu/CMakeLists.txt",
                "verification/cpu/consumer.py",
                "verification/cpu/consumer.cc",
                "scripts/verify_installed_consumers.py",
            ),
        )

    def check_recipe_lifecycle(self, group, names):
        # Exercise each input independently so one detected change cannot hide another.
        for name in names:
            with self.subTest(name=name):
                target = self.root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("original", encoding="utf-8")
                added = baseline.compare(self.expected, baseline.inventory(self.root))
                self.assertEqual(added[group]["added"], [name])
                previous = {**self.expected, "groups": baseline.inventory(self.root)}
                target.write_text("changed", encoding="utf-8")
                changed = baseline.compare(previous, baseline.inventory(self.root))
                self.assertEqual(changed[group]["changed"], [name])
                target.unlink()
                removed = baseline.compare(previous, baseline.inventory(self.root))
                self.assertEqual(removed[group]["removed"], [name])

    def test_capture_refuses_overwrite_and_check_exit_codes(self):
        path = self.root / "snapshot.json"
        command = [sys.executable, str(ROOT / "scripts/compatibility_baseline.py")]
        args = ["--root", str(self.root), "--baseline", str(path)]

        def run(mode, *extra):
            return subprocess.run(
                command + [mode] + args + list(extra), capture_output=True
            ).returncode

        self.assertEqual(run("capture", "--source-revision", "a" * 40), 0)
        self.assertEqual(run("capture", "--source-revision", "a" * 40), 2)
        self.assertEqual(run("check"), 0)
        (self.root / "src/base/add.h").write_text("changed", encoding="utf-8")
        self.assertEqual(run("check"), 1)


if __name__ == "__main__":
    unittest.main()
