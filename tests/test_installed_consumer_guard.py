"""Failure-path checks for the installed consumer lane, without runtime stubs."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "consumer_guard", ROOT / "scripts/verify_installed_consumers.py"
)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class ConsumerGuardTests(unittest.TestCase):
    def test_loaded_library_identity_is_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prefix, rt, foreign = (root / name for name in ("prefix", "rt", "foreign"))
            for directory in (prefix, rt, foreign):
                directory.mkdir()
            core = prefix / "libinfiniops.so"
            runtime = rt / "libinfinirt.so"
            core.write_bytes(b"core")
            runtime.write_bytes(b"runtime")
            copied = prefix / "libinfinirt.so"
            copied.write_bytes(b"runtime")
            self.assertTrue(guard.validate_loaded_libraries([core, copied], prefix, rt))
            wrong = foreign / "libinfiniops.so"
            wrong.write_bytes(b"core")
            for paths in ([core], [], [wrong, runtime]):
                with self.subTest(paths=paths), self.assertRaises(ValueError):
                    guard.validate_loaded_libraries(paths, prefix, rt)
            copied.write_bytes(b"wrong-runtime")
            with self.assertRaises(ValueError):
                guard.validate_loaded_libraries([core, copied], prefix, rt)

    def xml(self, body, required=("required",)):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cases.xml"
            path.write_text(body, encoding="utf-8")
            return guard.validate_junit(path, required)

    def test_required_actual_case_passes(self):
        self.assertEqual(
            self.xml('<testsuite tests="1"><testcase name="required"/></testsuite>'),
            ["required"],
        )

    def test_missing_zero_duplicate_or_invalid_root_fails(self):
        for xml in (
            "<testsuite/>",
            '<testsuite><testcase name="other"/></testsuite>',
            '<testsuite><testcase name="required"/><testcase name="required"/></testsuite>',
            '<other><testcase name="required"/></other>',
            '<testsuite tests="2"><testcase name="required"/></testsuite>',
        ):
            with self.subTest(xml=xml), self.assertRaises(ValueError):
                self.xml(xml)

    def test_skipped_failed_or_errored_case_fails(self):
        for status in ("skipped", "failure", "error"):
            with self.subTest(status=status), self.assertRaises(ValueError):
                self.xml(
                    f'<testsuite><testcase name="required"><{status}/></testcase></testsuite>'
                )

    def test_suite_failure_and_empty_required_fail(self):
        for attrs in ('failures="1"', 'errors="1"', 'skipped="1"'):
            with self.subTest(attrs=attrs), self.assertRaises(ValueError):
                self.xml(f'<testsuite {attrs}><testcase name="required"/></testsuite>')
        with self.assertRaises(ValueError):
            self.xml("<testsuite/>", ())

    def test_source_output_and_reused_output_rejected(self):
        with self.assertRaises(ValueError):
            guard.verify(Path("/prefix"), Path("/rt"), ROOT / "build/evidence")
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileExistsError):
                guard.verify(Path("/prefix"), Path("/rt"), Path(temp))

    def test_missing_installation_writes_failure_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "evidence"
            with (
                patch.object(guard.platform, "system", return_value="Linux"),
                patch.object(guard.platform, "machine", return_value="x86_64"),
                self.assertRaises(ValueError),
            ):
                guard.verify(Path(temp) / "missing", Path(temp) / "rt", output)
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertIn("Missing installed header", report["error"])
            self.assertEqual(report["commands"], [])


if __name__ == "__main__":
    unittest.main()
