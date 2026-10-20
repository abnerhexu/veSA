"""Serial, memory-bounded Q1 area and cycle experiments; no P&R/STA."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DESIGNS = ("ieee", "hopper", "blackwell", "hopper-overlap", "blackwell-overlap",
           "hopper-reuse-overlap", "blackwell-reuse-overlap")
PDK_VERSION = "c6d73a35f524070e85faff4a6a9eef49553ebc2b"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("all", "vectors", "rtl", "simulate", "synth", "plot"))
    p.add_argument("--output", type=Path)
    p.add_argument("--design", choices=(*DESIGNS, "overlap", "reuse", "q3"), help="Select one design, an overlap/reuse pair, or IEEE and both reuse designs")
    p.add_argument("--dim", type=int, choices=(2, 4, 8, 16), default=2)
    p.add_argument("--problem-dim", type=int, choices=(2, 4, 8, 16), help="Fixed global square task, serialized into array-sized output tiles")
    p.add_argument("--suite", type=Path, default=Path("/home/hexu/tool/oss-cad-suite"))
    p.add_argument("--pdk-root", type=Path, default=Path.home() / ".volare")
    args = p.parse_args()
    if args.dim != 2 and args.problem_dim is None:
        args.problem_dim = args.dim
    if args.problem_dim is not None and args.problem_dim < args.dim:
        p.error("--problem-dim must be at least --dim")
    out = (args.output or ROOT / "out/dac-q1" / datetime.now().strftime("%Y%m%d-%H%M%S")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.design == "q3":
        designs = ["ieee", "hopper-reuse-overlap", "blackwell-reuse-overlap"]
    elif args.design in ("overlap", "reuse"):
        designs = [d for d in DESIGNS if d.endswith("-overlap") and (("-reuse-" in d) == (args.design == "reuse"))]
    else:
        designs = [args.design] if args.design else DESIGNS
    lib = args.pdk_root / "volare/sky130/versions" / PDK_VERSION / "sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"
    env = dict(os.environ, JAVA_TOOL_OPTIONS="-Xmx1536m -XX:ActiveProcessorCount=1",
               MAKEFLAGS="-j1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               TENSORCORE_SIM="verilator", Q1_VECTORS=str(out / "vectors.tsv"), Q1_OUTPUT=str(out),
               Q1_DESIGNS=",".join(designs),
               Q1_DIM=str(args.dim), Q1_PROBLEM_DIM=str(args.problem_dim or args.dim),
               MPLCONFIGDIR=str(out / "mpl-cache"))
    mill = ["mill", "--no-server", "-j", "1"]
    inputs = [ROOT / "build.mill", ROOT / "integration/dependencies.lock",
              ROOT / "modeling/tensorcore/reference.py",
              *sorted((ROOT / "experiments/dac").rglob("*.scala")),
              *sorted((ROOT / "experiments/dac").rglob("*.py")),
              *sorted((ROOT / "src/main/scala/gemmini/tensorcore").glob("*.scala"))]
    inputs = [f for f in inputs if "__pycache__" not in f.parts]
    manifest = {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "source_sha256": {str(f.relative_to(ROOT)): sha(f) for f in inputs},
                "dirty_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)).hexdigest(),
                "pdk_version": PDK_VERSION, "liberty": str(lib), "liberty_sha256": sha(lib),
                "scope": f"{args.dim}x{args.dim} E4M3 core-only; no SRAM/DMA; no STA; native SKY130 area",
                "array_dim": args.dim, "problem_dim": args.problem_dim or args.dim,
                "memory_limit": "JVM 1536 MiB; Yosys/ABC each 4 GiB virtual address space; serial tasks",
                "versions": {name: importlib.metadata.version(name) for name in ("torch", "mmasim", "matplotlib")},
                "yosys_version": subprocess.check_output([str(args.suite / "bin/yosys"), "-V"], text=True).strip()}
    runs_path = out / "runs.csv"
    runs = list(csv.DictReader(runs_path.open())) if runs_path.exists() else []
    manifest_name = f"manifest-{len(runs):02d}.json"
    if (out / "vectors.tsv").exists():
        manifest["vectors_sha256"] = sha(out / "vectors.tsv")
    def save_manifest():
        payload = json.dumps(manifest, indent=2) + "\n"
        (out / manifest_name).write_text(payload)
        (out / "manifest.json").write_text(payload)
    save_manifest()
    for row in runs:
        row.setdefault("manifest", "manifest-00.json")
    print(f"Output: {out}", flush=True)

    def run(label, argv, bounded=False):
        log = out / f"{len(runs):02d}-{label}.log"
        usage = log.with_suffix(".resources.txt")
        def limit():
            resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
        started = time.monotonic()
        print(f"Running {label}", flush=True)
        with log.open("w") as stream:
            proc = subprocess.run(["/usr/bin/time", "-v", "-o", str(usage), *map(str, argv)],
                                  cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                  preexec_fn=limit if bounded else None)
        rss = next((line.split(":", 1)[1].strip() for line in usage.read_text().splitlines()
                    if "Maximum resident set size" in line), "")
        runs.append({"run_id": f"{out.name}-{len(runs):02d}", "stage": label,
                     "status": "passed" if proc.returncode == 0 else "failed",
                     "seconds": round(time.monotonic()-started, 3), "max_rss_kib": rss,
                     "exit_code": proc.returncode, "log": log.name, "command": json.dumps(list(map(str, argv))),
                     "manifest": manifest_name})
        write_csv(runs_path, runs)
        print(f"{label}: {runs[-1]['status']}, peak RSS {rss} KiB", flush=True)
        if proc.returncode:
            print("\n".join(log.read_text().splitlines()[-35:]), flush=True)
            raise SystemExit(proc.returncode)

    if args.stage in ("all", "vectors"):
        argv = [sys.executable, ROOT / "experiments/dac/vectors.py", out / "vectors.tsv", args.dim]
        if args.problem_dim is not None:
            argv.append(args.problem_dim)
        run("vectors", argv)
        manifest["vectors_sha256"] = sha(out / "vectors.tsv")
        save_manifest()
    if args.stage in ("all", "rtl"):
        for design in designs:
            run(f"rtl-{design}", mill + ["dac.runMain", "gemmini.dac.GenerateQ1", design, out / f"rtl-{design}", args.dim])
    if args.stage in ("all", "simulate"):
        argv = mill + ["dac.test.testOnly", "gemmini.dac.Q1Spec"]
        if args.design and args.design != "q3":
            argv += ["--", "-z", "-overlap" if args.design == "overlap" else args.design]
        run("simulate", argv)
        runtime = []
        for design in DESIGNS:
            path = out / f"{design}-runtime.csv"
            if path.exists():
                runtime.extend(csv.DictReader(path.open()))
        write_csv(out / "runtime.csv", runtime)
        tasks = []
        for design in designs:
            tasks.extend(csv.DictReader((out / f"{design}-task-runtime.csv").open()))
        write_csv(out / "task-runtime.csv", tasks)
    if args.stage in ("all", "synth"):
        for design in designs:
            # Flatten each PE for cross-boundary constant propagation, retain the
            # PE hierarchy for area accounting. No external SRAM in this scope.
            pe = "OverlapPE" if design.endswith("-overlap") else "IeeeFmaPE" if design == "ieee" else "TensorPE"
            forbidden = "-dont_use sky130_fd_sc_hd__lpflow_*"
            ys = out / f"{design}.ys"
            ys.write_text("\n".join([
                f"read_verilog -sv {out}/rtl-{design}/Q1Mesh.sv",
                "hierarchy -check -top Q1Mesh", f"flatten {pe}", "select *", "chformal -remove",
                "synth -top Q1Mesh -noabc", f"dfflibmap -liberty {lib} {forbidden}",
                f"abc -liberty {lib} {forbidden}", "clean",
                f"tee -o {out}/{design}-stat.json stat -json -liberty {lib}",
                f"write_verilog -noattr {out}/{design}-mapped.v"]) + "\n")
            run(f"synth-{design}", [args.suite / "bin/yosys", "-Q", "-T", "-s", ys], bounded=True)
        ppa, breakdown = [], []
        for design in DESIGNS:
            path = out / f"{design}-stat.json"
            if not path.exists():
                continue
            stats = json.loads(path.read_text())
            total = stats["design"]["area"]
            pe = "OverlapPE" if design.endswith("-overlap") else "IeeeFmaPE" if design == "ieee" else "TensorPE"
            profile = design.split("-")[0]
            pe_area = stats["modules"]["\\" + pe]["area"] * args.dim**2
            ppa.append({"design": design, "profile": "e4m3-f32", "rows": args.dim, "cols": args.dim,
                        "L": 32, "F": {"ieee": "", "hopper": 13, "blackwell": 25}[profile],
                        "rho": {"ieee": "RNE-FP32-per-FMA", "hopper": "RZ-E8M13", "blackwell": "RZ-FP32"}[profile],
                        "area_um2": total, "sequential_area_um2": stats["design"]["sequential_area"],
                        "source_node_nm": 130, "voltage": 1.8,
                        "temperature_c": 25, "memory_scope": "core-only", "frequency_basis": "not-assumed",
                        "flow_stage": "post-synthesis", "liberty_sha256": sha(lib)})
            for name, area in [("PE arithmetic and local state", pe_area), ("mesh control and forwarding", total-pe_area)]:
                breakdown.append({"design": design, "module_class": name, "area_um2": area})
        write_csv(out / "ppa.csv", ppa)
        write_csv(out / "breakdown.csv", breakdown)
    if args.stage in ("all", "plot"):
        run("plot", [sys.executable, ROOT / "experiments/dac/plot.py", out])


if __name__ == "__main__":
    main()
