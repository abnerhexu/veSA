"""Prepare or run the serial WS width/K area-power experiment handoff."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "modeling/tensorcore"))
from reference import dot  # noqa: E402

WIDTHS = (2, 4, 8, 16, 32)
KS = (32, 64, 128, 256)
ROWS = 16
PHYSICAL_DEPTH = 16
MAX_K = 256


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def first_csv_row(path):
    with Path(path).open() as stream:
        return next(csv.DictReader(stream))


def finite_e4m3(rng):
    if rng.random() < 0.03:
        return rng.choice((0, 0x80))
    return (rng.randrange(2) << 7) | (rng.randrange(2, 14) << 3) | rng.randrange(8)


def vector(path, columns, k, seed):
    rng = random.Random(f"ws-area:{seed}:{columns}:{k}")
    a = [[finite_e4m3(rng) for _ in range(k)] for _ in range(ROWS)]
    b = [[finite_e4m3(rng) for _ in range(columns)] for _ in range(k)]
    c = [[0 for _ in range(columns)] for _ in range(ROWS)]
    d = [[dot(a[row], [b[index][column] for index in range(k)], c[row][column])
          for column in range(columns)] for row in range(ROWS)]
    encode = lambda values: ",".join(f"{value:x}" for value in values)
    flatten = lambda matrix: [value for row in matrix for value in row]
    path.write_text("# checked E4M3/FP32 activity vector\n" +
                    f"a {encode(flatten(a))}\n" + f"b {encode(flatten(b))}\n" +
                    f"c {encode(flatten(c))}\n" + f"d {encode(flatten(d))}\n")


def manifest(out, seed):
    out.mkdir(parents=True, exist_ok=True)
    hardware = [dict(name=f"ws16x{columns}", physical_depth=PHYSICAL_DEPTH,
                     rows=ROWS, columns=columns, max_k=MAX_K,
                     physical_pes=PHYSICAL_DEPTH * columns,
                     input_product_lanes=PHYSICAL_DEPTH * columns,
                     numerical_fusion_length=32, operand_passes=1,
                     segment_buffers=2, weights_per_pe=4)
                for columns in WIDTHS]
    workloads = []
    for columns in WIDTHS:
        for k in KS:
            point = out / f"ws16x{columns}" / f"k{k}"
            point.mkdir(parents=True, exist_ok=True)
            vector_path = point / "vector.tsv"
            vector(vector_path, columns, k, seed)
            workloads.append(dict(hardware=f"ws16x{columns}", columns=columns,
                                  rows=ROWS, physical_depth=PHYSICAL_DEPTH, k=k,
                                  operations=2 * ROWS * columns * k,
                                  vector=str(vector_path.relative_to(out)),
                                  vector_sha256=sha(vector_path)))
    table(out / "hardware.csv", hardware)
    table(out / "workloads.csv", workloads)
    inputs = [ROOT / "build.mill", HERE / "prepare_area.py",
              HERE / "scala/WSReuseExperimentMesh.scala",
              HERE / "scala/GenerateWSReuse.scala",
              HERE / "test/WSReusePowerSpec.scala",
              ROOT / "src/main/scala/gemmini/tensorcore/TensorArithmetic.scala"]
    payload = dict(base_commit="f197fdfc52118d654740ac747134e207bf0483e7",
                   physical_depth=PHYSICAL_DEPTH, rows=ROWS, widths=list(WIDTHS),
                   k_values=list(KS), max_k=MAX_K, format="e4m3-f32",
                   scope="core-only; left-edge A buffering; no SRAM/DMA",
                   frequency_hz=100_000_000,
                   frequency_basis="assumed for power/throughput; no STA claim",
                   overlap_definition="cycles with next-segment exponent lane and current-segment sum lane both active",
                   throughput_definition="2*M*N*K / measured request-to-result cycles",
                   power_definition="OpenSTA pre-layout activity estimate over checked request-to-result VCD window",
                   source_sha256={str(path.relative_to(ROOT)): sha(path) for path in inputs},
                   hardware_points=len(hardware), workload_points=len(workloads), seed=seed)
    (out / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def run(argv, env=None):
    print("RUN", " ".join(map(str, argv)), flush=True)
    subprocess.run(list(map(str, argv)), cwd=ROOT, env=env, check=True)


def scale_vcd(source, destination, factor=5000):
    """Convert Chiseltest's 2 ps clock period to the declared 10 ns period."""
    with source.open() as incoming, destination.open("w") as outgoing:
        for line in incoming:
            if line.startswith("#"):
                line = f"#{int(line[1:]) * factor}\n"
            outgoing.write(line)


def generate_rtl(out, widths):
    for columns in widths:
        target = out / f"ws16x{columns}" / "rtl"
        run(("mill", "--no-server", "-j", "1", "wsExperiments.runMain",
             "gemmini.wsexperiments.GenerateWSReuse", PHYSICAL_DEPTH, ROWS,
             columns, MAX_K, target))


def simulate(out, widths, ks):
    env = dict(os.environ, TENSORCORE_SIM="verilator", MAKEFLAGS="-j4",
               JAVA_TOOL_OPTIONS="-Xmx1536m -XX:ActiveProcessorCount=1")
    for columns in widths:
        for k in ks:
            point = out / f"ws16x{columns}" / f"k{k}"
            env.update(WS_SWEEP_COLUMNS=str(columns), WS_SWEEP_K=str(k),
                       WS_SWEEP_VECTOR=str((point / "vector.tsv").resolve()),
                       WS_SWEEP_RUNTIME=str((point / "runtime.csv").resolve()))
            run(("mill", "--no-server", "-j", "1", "wsExperiments.test.testOnly",
                 "gemmini.wsexperiments.WSReusePowerSpec"), env)
            candidates = sorted((ROOT / "out/wsExperiments/test/testOnly.dest").rglob("*.vcd"),
                                key=lambda path: path.stat().st_mtime)
            if not candidates:
                raise RuntimeError("Chiseltest did not emit a VCD")
            shutil.copy2(candidates[-1], point / "activity-raw-2ps.vcd")
            scale_vcd(candidates[-1], point / "activity.vcd")


def synthesize(out, yosys, liberty, widths):
    for columns in widths:
        root = out / f"ws16x{columns}"
        rtl = root / "rtl/WSReuseExperimentMesh.sv"
        script = root / "synth.ys"
        script.write_text("\n".join((
            f"read_verilog -sv {rtl}",
            "hierarchy -check -top WSReuseExperimentMesh",
            "select *", "chformal -remove", "synth -top WSReuseExperimentMesh -noabc",
            f"dfflibmap -liberty {liberty} -dont_use sky130_fd_sc_hd__lpflow_*",
            f"abc -liberty {liberty} -dont_use sky130_fd_sc_hd__lpflow_*", "clean",
            f"tee -o {root/'stat.json'} stat -json -liberty {liberty}",
            f"write_verilog -noattr {root/'mapped.v'}")) + "\n")
        run((yosys, "-Q", "-T", "-s", script))


def power(out, openroad, liberty, tech_lef, cell_lef, widths, ks):
    for columns in widths:
        root = out / f"ws16x{columns}"
        for k in ks:
            point = root / f"k{k}"
            runtime = first_csv_row(point / "runtime.csv")
            # Scaled VCD uses 1 ps units and a 10,000 ps full clock cycle.
            begin = int(runtime["begin_cycle"]) * 10_000
            end = int(runtime["result_cycle"]) * 10_000
            script = point / "power.tcl"
            script.write_text("\n".join((
                f"read_lef {tech_lef}", f"read_lef {cell_lef}",
                f"read_liberty {liberty}", f"read_verilog {root/'mapped.v'}",
                "link_design WSReuseExperimentMesh",
                "create_clock -name clk -period 10 [get_ports clock]",
                "set_input_transition 0.1 [all_inputs]",
                "set_load 0.01 [all_outputs]",
                f"read_vcd -scope TOP/WSReuseExperimentMesh -begin_time {begin} -end_time {end} {point/'activity.vcd'}",
                f"report_activity_annotation > {point/'annotation.txt'}",
                f"report_power -digits 9 -format json > {point/'power.json'}",
                f"report_power -digits 9 > {point/'power.txt'}", "exit")) + "\n")
            run((openroad, "-exit", script))


def collect(out, widths, ks):
    areas = {}
    for columns in widths:
        stat = out / f"ws16x{columns}/stat.json"
        if not stat.exists():
            raise FileNotFoundError(f"missing synthesized area: {stat}")
        areas[columns] = json.loads(stat.read_text())["design"]["area"]
    rows = []
    for columns in widths:
        for k in ks:
            point = out / f"ws16x{columns}/k{k}"
            runtime = point / "runtime.csv"
            if not runtime.exists():
                raise FileNotFoundError(f"missing checked runtime: {runtime}")
            row = first_csv_row(runtime)
            if row.get("result_check") != "passed":
                raise RuntimeError(f"failed result check: {runtime}")
            cycles = int(row["cycles"]); operations = int(row["operations"])
            area = areas.get(columns, "")
            power_path = out / f"ws16x{columns}/k{k}/power.json"
            annotation = point / "annotation.txt"
            if not power_path.exists() or not annotation.exists():
                raise FileNotFoundError(f"missing power/annotation output for {point}")
            if not annotation.read_text().strip():
                raise RuntimeError(f"empty activity annotation: {annotation}")
            total_power = json.loads(power_path.read_text())["Total"]["total"]
            energy = float(total_power) * cycles / 100_000_000
            throughput_ops_s = operations / cycles * 100_000_000
            area_mm2 = float(area) / 1e6
            rows.append(dict(columns=columns, k=k, cycles=cycles,
                             operations=operations,
                             operations_per_cycle=operations / cycles,
                             area_um2=area,
                             throughput_ops_s=throughput_ops_s,
                             throughput_tops=throughput_ops_s / 1e12,
                             operations_per_cycle_per_mm2=(operations / cycles) / area_mm2,
                             throughput_tops_per_mm2=throughput_ops_s / 1e12 / area_mm2,
                             power_w=total_power, energy_j=energy,
                             power_density_w_per_mm2=float(total_power) / area_mm2,
                             energy_per_operation_j=float(energy) / operations,
                             overlap_cycles=row["overlap_cycles"],
                             activity_annotation="present",
                             note="pre-layout activity estimate; assumed 100 MHz; no STA"))
    table(out / "summary.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "rtl", "simulate", "synth", "power", "collect"))
    parser.add_argument("--output", type=Path, default=ROOT / "out/ws-sweeps-area")
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--columns", type=int, choices=WIDTHS,
                        help="Run one width; default runs all five")
    parser.add_argument("--k", type=int, choices=KS,
                        help="Run one workload K; default runs all four")
    parser.add_argument("--yosys", type=Path)
    parser.add_argument("--openroad", type=Path)
    parser.add_argument("--liberty", type=Path)
    parser.add_argument("--tech-lef", type=Path)
    parser.add_argument("--cell-lef", type=Path)
    args = parser.parse_args(); out = args.output.resolve()
    widths = (args.columns,) if args.columns else WIDTHS
    ks = (args.k,) if args.k else KS
    if args.stage == "prepare": manifest(out, args.seed)
    else:
        if not (out / "manifest.json").exists():
            manifest(out, args.seed)
        if args.stage == "rtl": generate_rtl(out, widths)
        elif args.stage == "simulate": simulate(out, widths, ks)
        elif args.stage == "synth":
            if args.yosys is None or args.liberty is None:
                parser.error("synth requires --yosys and --liberty")
            synthesize(out, args.yosys.resolve(), args.liberty.resolve(), widths)
        elif args.stage == "power":
            if any(value is None for value in
                   (args.openroad, args.liberty, args.tech_lef, args.cell_lef)):
                parser.error("power requires --openroad, --liberty, --tech-lef and --cell-lef")
            power(out, args.openroad.resolve(), args.liberty.resolve(),
                  args.tech_lef.resolve(), args.cell_lef.resolve(), widths, ks)
        else: collect(out, widths, ks)


if __name__ == "__main__":
    main()
