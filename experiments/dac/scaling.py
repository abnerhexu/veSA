"""Compare measured array sizes on identical global GEMM tasks."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_array_scaling(output, points, problem):
    dims = sorted({int(p["array_dim"]) for p in points})
    ks = sorted({int(p["K"]) for p in points})
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.8), layout="constrained")
    for design, label, color, marker in [("ieee", "IEEE FMA", "#666666", "o"),
                                        ("hopper-reuse-overlap", "Hopper reuse", "#D55E00", "s"),
                                        ("blackwell-reuse-overlap", "Blackwell reuse", "#0072B2", "^")]:
        rows = sorted([p for p in points if p["design"] == design and p["K"] == "256"], key=lambda p: int(p["array_dim"]))
        for ax, field in zip(axes, ("operations_per_cycle", "density_ratio_vs_2x2")):
            ax.plot(dims, [p[field] for p in rows], color=color, marker=marker,
                    markersize=5 if design == "hopper-reuse-overlap" else 3.5, label=label)
    for ax, title, ylabel in zip(axes, (f"(a) M=N={problem}, K=256", "(b) Density, K=256"),
                                 ("Operations / cycle", "Density / same-design 2×2")):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Array dimension")
        ax.set_xscale("log", base=2)
        ax.set_xticks(dims, [str(d) for d in dims])
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[1].axhline(1, color="#888888", linewidth=.6)
    axes[1].set_ylim(top=1.1*max(p["density_ratio_vs_2x2"] for p in points if p["K"] == "256"))
    occupancy = [[next(p["effective_mac_slot_fraction"] for p in points
                       if p["design"] == "blackwell-reuse-overlap" and int(p["array_dim"]) == d and int(p["K"]) == k)
                  for k in ks] for d in dims]
    im = axes[2].imshow(occupancy, vmin=0, vmax=1, cmap="cividis", aspect="auto")
    axes[2].set_title("(c) Blackwell MAC slots")
    axes[2].set_xticks(range(len(ks)), [str(k) for k in ks], rotation=45)
    axes[2].set_yticks(range(len(dims)), [str(d) for d in dims])
    axes[2].set_xlabel("K")
    axes[2].set_ylabel("Array dimension")
    fig.colorbar(im, ax=axes[2], fraction=.05, pad=.04)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False, fontsize=7)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output / f"q3-array-scaling.{suffix}", dpi=300)
    plt.close(fig)


def main(output, directories):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    points = []
    inputs = {}
    for directory in map(Path, directories):
        manifest = json.loads((directory / "manifest.json").read_text())
        dim, problem = manifest["array_dim"], manifest["problem_dim"]
        # Reassemble raw A/B/C and all three expected results to establish that
        # every array size executes exactly the same global tasks and outputs.
        tasks = {}
        for line in (directory / "vectors.tsv").read_text().splitlines():
            if line.startswith("#"):
                continue
            v = line.split()
            name, offset = v[0].split("_tile")
            row, col = map(int, offset.split("_"))
            k = int(v[1])
            arrays = [[int(x, 16) for x in field.split(",")] for field in v[2:]]
            task = tasks.setdefault(name, [[None]*(problem*k), [None]*(k*problem),
                                          *[[None]*(problem**2) for _ in range(4)]])
            task[0][row*k:(row+dim)*k] = arrays[0]
            for t in range(k):
                task[1][t*problem+col:t*problem+col+dim] = arrays[1][t*dim:(t+1)*dim]
            for field in range(2, 6):
                for r in range(dim):
                    start = (row+r)*problem+col
                    task[field][start:start+dim] = arrays[field][r*dim:(r+1)*dim]
        digest = hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest()
        inputs[str(directory)] = {"global_inputs_and_expected_sha256": digest,
                                  "vectors_sha256": hashlib.sha256((directory / "vectors.tsv").read_bytes()).hexdigest(),
                                  "ppa_sha256": hashlib.sha256((directory / "ppa.csv").read_bytes()).hexdigest(),
                                  "task_runtime_sha256": hashlib.sha256((directory / "task-runtime.csv").read_bytes()).hexdigest(),
                                  "manifest": str(directory / "manifest.json")}
        areas = {r["design"]: float(r["area_um2"]) for r in csv.DictReader((directory / "ppa.csv").open())}
        for row in csv.DictReader((directory / "task-runtime.csv").open()):
            if not row["workload"].startswith("gemm_"):
                continue
            assert row["result_check"] == "passed"
            cycles = int(row["cycles"])
            m, n, k = (int(row[key]) for key in ("M", "N", "K"))
            area = areas[row["design"]]
            points.append({**row, "area_um2": area, "operations_per_cycle": 2*m*n*k/cycles,
                           "effective_mac_slot_fraction": m*n*k/(dim*dim*cycles),
                           "operations_per_cycle_per_mm2": 2*m*n*k/cycles/area*1e6})
    assert len({v["global_inputs_and_expected_sha256"] for v in inputs.values()}) == 1
    for p in points:
        base = next(b for b in points if b["design"] == p["design"] and b["K"] == p["K"] and b["array_dim"] == "2")
        p["speedup_vs_2x2"] = int(base["cycles"]) / int(p["cycles"])
        p["area_ratio_vs_2x2"] = p["area_um2"] / base["area_um2"]
        p["density_ratio_vs_2x2"] = p["speedup_vs_2x2"] / p["area_ratio_vs_2x2"]
    with (output / "scaling.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    (output / "sources.json").write_text(json.dumps({
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs": inputs}, indent=2) + "\n")
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    if len({p["array_dim"] for p in points}) > 2:
        plot_array_scaling(output, points, problem)
        print(f"Saved {len(points)} measured points for global M=N={problem}; global input/expected hash: {digest}")
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.7), layout="constrained")
    for design, label, color, marker in [("ieee", "IEEE FMA", "#666666", "o"),
                                        ("hopper-reuse-overlap", "Hopper reuse", "#D55E00", "s"),
                                        ("blackwell-reuse-overlap", "Blackwell reuse", "#0072B2", "^")]:
        for dim in sorted({int(p["array_dim"]) for p in points}):
            rows = sorted([p for p in points if p["design"] == design and int(p["array_dim"]) == dim], key=lambda r: int(r["K"]))
            for ax, field in zip(axes, ("cycles", "speedup_vs_2x2", "density_ratio_vs_2x2")):
                ax.plot([int(r["K"]) for r in rows], [float(r[field]) for r in rows],
                        color=color, marker=marker, markersize=4, linestyle="--" if dim == 2 else "-",
                        markerfacecolor="none" if dim == 2 else color, label=f"{label} {dim}×{dim}")
    for ax, title, ylabel in zip(axes, ("(a) Fixed global task", "(b) Cycle speedup", "(c) Throughput density"),
                                 ("Request → final result (cycles)", "Speedup vs same-design 2×2", "Density / same-design 2×2")):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(f"K (global M=N={problem})")
        ax.set_xticks([64, 128, 256])
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    for ax in axes[1:]:
        ax.axhline(1, color="#888888", linewidth=.6)
    axes[1].set_ylim(top=max(p["speedup_vs_2x2"] for p in points)*1.1)
    axes[2].set_ylim(top=max(p["density_ratio_vs_2x2"] for p in points)*1.1)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False, fontsize=7)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output / f"q3-scaling.{suffix}", dpi=300)
    plt.close(fig)
    print(f"Saved {len(points)} measured points; identical reassembled global input/expected hash: {digest}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
