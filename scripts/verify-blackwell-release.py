"""Fail-closed Blackwell release checks and strictly sequential reproductions.

Run with the parent's .venv Python, from any working directory. No installation,
network fetch, golden-file replacement, source edit or Git mutation is performed.
Results/logs are written to a fresh ignored out/blackwell-release directory.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "integration/blackwell-release.json"
STAGES = ("check", "fixtures", "cpu", "core", "memory", "integration", "all")
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class CommandFailed(RuntimeError):
    def __init__(self, message, record):
        super().__init__(message)
        self.record = record


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(*args, cwd=ROOT):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def verify_hashes(base, hashes):
    for name, expected in hashes.items():
        path = base / name
        require(path.is_file(), f"Missing pinned file: {path}")
        require(sha256(path) == expected, f"SHA-256 mismatch: {path}")


def fixture_counts(base):
    pe, matrices = Counter(), Counter()
    for filename, grouped, counts in (
        ("vectors.tsv", False, pe), ("grouped-vectors.tsv", True, pe),
        ("matrices.tsv", False, matrices), ("grouped-matrices.tsv", True, matrices),
    ):
        for line in (base / filename).read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            counts[fields[0] if grouped else "-".join(fields[:2])] += 1
    return pe, matrices


def preflight(manifest):
    require(manifest["schema"] == 1, "Unsupported manifest schema")
    require(sys.flags.optimize == 0 and not os.environ.get("PYTHONOPTIMIZE"),
            "Python optimization disables assertions in the pinned author validators")
    verify_hashes(ROOT, manifest["sha256"])
    base = manifest["source_base"]
    paths = manifest["frozen_paths"]
    require(capture("git", "diff", "--name-only", base, "--", *paths) == "",
            "Frozen implementation/build/test paths differ from the validated source base")
    require(capture("git", "ls-files", "--others", "--exclude-standard", "--", *paths) == "",
            "Untracked files in frozen build/source paths could change the tested design")
    for package, version in manifest["versions"].items():
        require(importlib.metadata.version(package) == version,
                f"Expected {package}=={version}")
    author = importlib.metadata.distribution("mmasim")
    for name, expected in manifest["author_sha256"].items():
        require(sha256(Path(author.locate_file(name))) == expected,
                f"Pinned author implementation changed: {name}")
    pe, matrices = fixture_counts(ROOT / "modeling/tensorcore")
    require(dict(pe) == {p: v["pe_vectors"] for p, v in manifest["profiles"].items()},
            "PE fixture profile coverage/count changed")
    require(dict(matrices) == {p: v["matrices"] for p, v in manifest["profiles"].items()},
            "Matrix fixture profile coverage/count changed")
    for line in (ROOT / "integration/dependencies.lock").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, remote, pin = line.split()
        cache = ROOT / "out/integration-deps" / name
        require((cache / ".git").exists(), "Run scripts/setup-tensorcore-integration.sh first")
        require(capture("git", "remote", "get-url", "origin", cwd=cache) == remote,
                f"Unexpected dependency remote: {name}")
        require(capture("git", "rev-parse", "HEAD", cwd=cache) == pin,
                f"Dependency is not at pinned commit: {name}")
        require(capture("git", "status", "--porcelain", cwd=cache) == "",
                f"Dependency has local changes: {name}")
    # The submodule also works standalone; PDFs belong to the parent repository.
    papers = ROOT.parent / "docs"
    if all((papers / name).is_file() for name in manifest["papers_sha256"]):
        verify_hashes(papers, manifest["papers_sha256"])
    mill_version = capture("mill", "--no-server", "--version")
    require("Mill Build Tool version " + manifest["tools"]["mill"] + "\n" in mill_version + "\n",
            "Unexpected Mill version")
    verilator_version = capture(os.environ.get("TENSORCORE_VERILATOR_BIN", "verilator_bin"), "--version")
    require(verilator_version.startswith("Verilator " + manifest["tools"]["verilator"] + " "),
            "Unexpected Verilator version for the pinned chiseltest harness")
    return {"pe_vectors": sum(pe.values()), "matrices": sum(matrices.values()),
            "profiles": list(manifest["profiles"]), "source_base": base,
            "head": capture("git", "rev-parse", "HEAD"),
            "worktree": capture("git", "status", "--porcelain"),
            "manifest_sha256": sha256(MANIFEST), "python": sys.version,
            "mill_launcher": mill_version, "verilator": verilator_version}


@contextmanager
def exclusive(path):
    # Keep the inode; unlinking a lock could let a third process bypass a waiter.
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another Blackwell release runner is active; run one task at a time") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def check_scala_summary(output, expected):
    clean = ANSI.sub("", output)
    counts = re.findall(r"Total number of tests run:\s*(\d+)", clean)
    require(counts == [str(expected)] and "All tests passed." in clean,
            f"Expected exactly {expected} passing ScalaTests (zero-test filters must not pass)")


def commands(stage, run_dir, manifest):
    python = sys.executable
    mill = ["mill", "--no-server", "-j", "1"]
    def scala(name, count):
        return (name, mill + ["tensorcore.test.testOnly", "gemmini.tensorcore." + name], count)
    if stage == "fixtures":
        return [
            ("cpu-unit", [python, "-m", "unittest", "discover", "-s", "modeling/tensorcore", "-p", "test_*.py", "-v"], None),
            ("unscaled-export", [python, "modeling/tensorcore/validate.py", "--cases", "1000", "--vectors", str(run_dir / "vectors.tsv"), "--matrix-vectors", str(run_dir / "matrices.tsv")], None),
            ("grouped-export", [python, "modeling/tensorcore/validate_grouped.py", "--cases", "1000", "--vectors", str(run_dir / "grouped-vectors.tsv"), "--matrix-vectors", str(run_dir / "grouped-matrices.tsv")], None),
        ]
    if stage == "cpu":
        return [
            ("unscaled-million", [python, "modeling/tensorcore/validate.py", "--cases", "150000"], None),
            ("grouped-million", [python, "modeling/tensorcore/validate_grouped.py", "--cases", "500000"], None),
        ]
    if stage == "core":
        return [scala(name, count) for name, count in (
            ("BuildSmokeSpec", 1), ("TensorPESpec", 7), ("TensorMeshSpec", 7),
            ("GroupedFrontendSpec", 2), ("GroupedPESpec", 2), ("GroupedMeshSpec", 2))]
    if stage == "memory":
        return [scala(name, count) for name, count in (
            ("AccumulatorContractsSpec", 7), ("AccumulatorRouterSpec", 1),
            ("TensorExecuteControlSpec", 1), ("TensorCommandFrontendSpec", 3),
            ("TensorCommandIsolationSpec", 1))]
    if stage == "integration":
        script = "./scripts/test-tensorcore-integration.sh"
        return [("full-compile", [script, "compile"], None),
                ("rtl-enabled", [script, "rtl-e4m3"], None),
                ("rtl-disabled", [script, "rtl-disabled"], None)] + [
            ("rocc-" + profile, [script, profile], 1) for profile in manifest["profiles"]]
    return []


def run_command(label, argv, expected, directory, env):
    print(f"RUN {label}: {shlex.join(argv)}", flush=True)
    start = time.monotonic()
    log = directory / (label + ".log")
    # subprocess.run waits for the entire process tree's inherited stdout to
    # close. No background jobs, parallel futures, shell pipelines or resume skips.
    with log.open("w") as stream:
        result = subprocess.run(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    record = {"name": label, "command": argv, "returncode": result.returncode,
              "seconds": round(time.monotonic() - start, 3), "log": log.name,
              "log_sha256": sha256(log), "expected_tests": expected}
    try:
        require(result.returncode == 0, f"{label} failed ({result.returncode}); see {log}")
        if expected is not None:
            check_scala_summary(log.read_text(), expected)
    except RuntimeError as error:
        record["status"] = "failed"
        raise CommandFailed(str(error), record) from error
    record["status"] = "passed"
    print(f"PASS {label} ({record['seconds']} s): {log}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES, nargs="?", default="check")
    parser.add_argument("--list", action="store_true", help="only print commands; run nothing")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    selected = list(STAGES[1:-1]) if args.stage == "all" else [args.stage]
    if args.list:
        for stage in selected:
            for label, argv, _ in commands(stage, Path("RUN_DIR"), manifest):
                print(f"{stage}/{label}: {shlex.join(argv)}")
        return
    output = ROOT / "out/blackwell-release"
    output.mkdir(parents=True, exist_ok=True)
    with exclusive(output / "runner.lock"):
        directory = Path(tempfile.mkdtemp(prefix=args.stage + "-", dir=output))
        report = {"release": manifest["release"], "stage": args.stage,
                  "started_utc": datetime.now(timezone.utc).isoformat(),
                  "status": "running", "commands": []}
        print(f"Report: {directory / 'result.json'}", flush=True)
        env = dict(os.environ)
        env["JAVA_TOOL_OPTIONS"] = env.get("JAVA_TOOL_OPTIONS", "") + " -Xmx1536m -XX:ActiveProcessorCount=1"
        env.update(TENSORCORE_SIM="verilator", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   MAKEFLAGS="-j1", MILL_MAX_CONCURRENT_TASKS="1")
        try:
            report["preflight"] = preflight(manifest)
            for stage in selected:
                for label, argv, expected in commands(stage, directory, manifest):
                    report["commands"].append(run_command(label, argv, expected, directory, env))
                    (directory / "result.json").write_text(json.dumps(report, indent=2) + "\n")
                if stage == "fixtures":
                    verify_hashes(directory, {Path(n).name: h for n, h in manifest["sha256"].items()
                                              if n.endswith(".tsv")})
                    print("PASS byte-for-byte reproduction of all four pinned TSV files", flush=True)
            if "integration" in selected:
                disabled = (ROOT / "out/integration-rtl-disabled/GemminiIntegrationHarness.sv").read_text()
                require(not re.search(r"module\s+(TensorCommandFrontend|TensorExecute|TensorMesh)\b", disabled),
                        "Disabled configuration unexpectedly contains Tensor hardware")
            report["postflight"] = preflight(manifest)
            require({k: v for k, v in report["preflight"].items() if k != "worktree"} ==
                    {k: v for k, v in report["postflight"].items() if k != "worktree"},
                    "Source state changed during verification")
            report["status"] = "passed"
            print(f"PASS {args.stage}; this stage alone is not the entire release gate", flush=True)
        except Exception as error:
            if isinstance(error, CommandFailed):
                report["commands"].append(error.record)
            report["status"] = "failed"
            report["error"] = str(error)
            raise
        finally:
            report["finished_utc"] = datetime.now(timezone.utc).isoformat()
            (directory / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
