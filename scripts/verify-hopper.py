"""Serial Hopper development/regression gates, retaining Blackwell coverage.

The immutable Blackwell v1 verifier intentionally rejects this changed source
tree. This runner tests both profiles in the Hopper branch, without relaxing
that original release lock. Each stage retains logs and an honest JSON result.
"""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from datetime import datetime, timezone

spec = importlib.util.spec_from_file_location(
    "blackwell_gate", Path(__file__).with_name("verify-blackwell-release.py"))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
ROOT = gate.ROOT


def snapshot():
    files = [ROOT / "build.mill", ROOT / "integration/dependencies.lock"]
    for folder in ("src/main/scala", "src/test/scala/gemmini/tensorcore", "integration/src",
                   "modeling/tensorcore", "scripts"):
        files.extend(p for p in (ROOT / folder).rglob("*")
                     if p.is_file() and p.suffix in (".scala", ".py", ".sh", ".tsv"))
    return {str(p.relative_to(ROOT)): gate.sha256(p) for p in sorted(set(files))}


def preflight():
    manifest = json.loads(gate.MANIFEST.read_text())
    gate.require(sys.flags.optimize == 0 and not os.environ.get("PYTHONOPTIMIZE"), "Do not disable assertions")
    for package, version in manifest["versions"].items():
        gate.require(importlib.metadata.version(package) == version, f"Expected {package}=={version}")
    author = importlib.metadata.distribution("mmasim")
    for name, expected in manifest["author_sha256"].items():
        gate.require(gate.sha256(Path(author.locate_file(name))) == expected, f"Author file changed: {name}")
    gate.verify_hashes(ROOT, {n: h for n, h in manifest["sha256"].items() if n.endswith(".tsv")})
    return snapshot()


def plan(stage):
    mill = ["mill", "--no-server", "-j", "1"]
    def scala(name, count):
        return name, mill + ["tensorcore.test.testOnly", "gemmini.tensorcore." + name], count
    if stage == "cpu":
        return [
            ("unit", [sys.executable, "-m", "unittest", "discover", "-s", "modeling/tensorcore", "-p", "test_*.py", "-v"], None),
            ("hopper-million", [sys.executable, "modeling/tensorcore/validate.py", "--architecture", "hopper", "--cases", "150000"], None),
            ("blackwell-million", [sys.executable, "modeling/tensorcore/validate.py", "--cases", "150000"], None),
            ("blackwell-grouped-million", [sys.executable, "modeling/tensorcore/validate_grouped.py", "--cases", "500000"], None),
        ]
    if stage == "core":
        return [scala(n, c) for n, c in (("TensorProfilesSpec", 2), ("HopperPESpec", 7),
                ("HopperMeshSpec", 7), ("TensorPESpec", 7), ("TensorMeshSpec", 7),
                ("GroupedFrontendSpec", 2), ("GroupedPESpec", 2), ("GroupedMeshSpec", 2))]
    profiles = ["e4m3-f32", "e4m3-f16", "e5m2-f32", "e5m2-f16", "fp16-f32", "fp16-f16", "bf16-f32"]
    if stage == "integration":
        return [("compile", ["./scripts/test-tensorcore-hopper.sh", "compile"], None),
                ("rtl-e4m3", ["./scripts/test-tensorcore-hopper.sh", "rtl-e4m3"], None),
                ("rtl-fp16", ["./scripts/test-tensorcore-hopper.sh", "rtl-fp16"], None)] + [
            (p, ["./scripts/test-tensorcore-hopper.sh", "integration-" + p], 1) for p in profiles]
    return [(p, ["./scripts/test-tensorcore-integration.sh", p], 1) for p in profiles + ["mxfp4", "nvfp4"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("cpu", "core", "integration", "blackwell"))
    args = parser.parse_args()
    output = ROOT / "out/hopper-verification"
    output.mkdir(parents=True, exist_ok=True)
    # Same lock as the original release runner; manual tasks must still obey
    # the project-wide one-task rule.
    lock_dir = ROOT / "out/blackwell-release"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with gate.exclusive(lock_dir / "runner.lock"):
        directory = Path(tempfile.mkdtemp(prefix=args.stage + "-", dir=output))
        report = {"stage": args.stage, "status": "running", "commands": [],
                  "head": gate.capture("git", "rev-parse", "HEAD"),
                  "started_utc": datetime.now(timezone.utc).isoformat()}
        print(f"Report: {directory / 'result.json'}", flush=True)
        env = dict(os.environ)
        env["JAVA_TOOL_OPTIONS"] = env.get("JAVA_TOOL_OPTIONS", "") + " -Xmx1536m -XX:ActiveProcessorCount=1"
        env.update(TENSORCORE_SIM="verilator", MAKEFLAGS="-j1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        try:
            report["source_sha256"] = preflight()
            for name, argv, count in plan(args.stage):
                report["commands"].append(gate.run_command(name, argv, count, directory, env))
                (directory / "result.json").write_text(json.dumps(report, indent=2) + "\n")
            gate.require(preflight() == report["source_sha256"], "Verification inputs changed during stage")
            report["status"] = "passed"
            print(f"PASS {args.stage}", flush=True)
        except Exception as error:
            if isinstance(error, gate.CommandFailed):
                report["commands"].append(error.record)
            report.update(status="failed", error=str(error))
            raise
        finally:
            report["finished_utc"] = datetime.now(timezone.utc).isoformat()
            (directory / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
