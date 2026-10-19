"""Fast standard-library tests for the release gate; no simulator is started."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "blackwell_release", Path(__file__).with_name("verify-blackwell-release.py"))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class ReleaseGateTests(unittest.TestCase):
    def test_passing_scala_summary(self):
        gate.check_scala_summary("\x1b[32mTotal number of tests run: 7\x1b[0m\nAll tests passed.", 7)

    def test_zero_tests_are_not_success(self):
        with self.assertRaisesRegex(RuntimeError, "exactly 1"):
            gate.check_scala_summary("Total number of tests run: 0\nAll tests passed.", 1)

    def test_wrong_or_missing_summary_is_not_success(self):
        for output in ("", "Total number of tests run: 2\nAll tests passed.",
                       "Total number of tests run: 1\nA TEST FAILED",
                       "Total number of tests run: 1\nTotal number of tests run: 1\nAll tests passed."):
            with self.subTest(output=output), self.assertRaises(RuntimeError):
                gate.check_scala_summary(output, 1)

    def test_hash_mismatch_and_missing_file_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            base = Path(name)
            path = base / "vector.tsv"
            path.write_text("original\n")
            hashes = {path.name: gate.sha256(path)}
            gate.verify_hashes(base, hashes)
            path.write_text("altered\n")
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                gate.verify_hashes(base, hashes)
            with self.assertRaisesRegex(RuntimeError, "Missing"):
                gate.verify_hashes(base, {"missing.tsv": hashes[path.name]})

    def test_lock_rejects_overlap_and_is_reusable(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "lock"
            with gate.exclusive(path):
                with self.assertRaisesRegex(RuntimeError, "Another Blackwell"):
                    with gate.exclusive(path):
                        self.fail("Second runner entered")
            with gate.exclusive(path):
                self.assertTrue(path.exists())

    def test_lock_released_on_failure(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "lock"
            with self.assertRaises(ValueError):
                with gate.exclusive(path):
                    raise ValueError("test")
            with gate.exclusive(path):
                pass

    def test_exact_nine_profile_plan_and_fixture_counts(self):
        manifest = json.loads(gate.MANIFEST.read_text())
        expected = {"e4m3-f32", "e4m3-f16", "e5m2-f32", "e5m2-f16",
                    "fp16-f32", "fp16-f16", "bf16-f32", "mxfp4", "nvfp4"}
        self.assertEqual(set(manifest["profiles"]), expected)
        plan = gate.commands("integration", Path("unused"), manifest)
        self.assertEqual({argv[1] for _, argv, n in plan if n == 1}, expected)
        self.assertEqual(sum(n == 1 for _, _, n in plan), 9)
        pe, matrices = gate.fixture_counts(gate.ROOT / "modeling/tensorcore")
        self.assertEqual(dict(pe), {p: v["pe_vectors"] for p, v in manifest["profiles"].items()})
        self.assertEqual(dict(matrices), {p: v["matrices"] for p, v in manifest["profiles"].items()})
        self.assertEqual(sum(pe.values()), 1267)
        self.assertEqual(sum(matrices.values()), 96)

    def test_command_failures_are_not_success(self):
        with tempfile.TemporaryDirectory() as name, redirect_stdout(io.StringIO()):
            directory = Path(name)
            with self.assertRaisesRegex(gate.CommandFailed, "failed \\(23\\)") as raised:
                gate.run_command("reject", [gate.sys.executable, "-c", "raise SystemExit(23)"],
                                 None, directory, dict(gate.os.environ))
            self.assertEqual(raised.exception.record["returncode"], 23)
            self.assertEqual(raised.exception.record["status"], "failed")
            self.assertTrue((directory / "reject.log").is_file())


if __name__ == "__main__":
    unittest.main()
