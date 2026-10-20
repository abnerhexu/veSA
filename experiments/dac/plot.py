"""Plot measured Q1 CSV tables; no interpolated or projected data."""
import csv
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def main(directory):
    root = Path(directory)
    ppa = {row["design"]: row for row in csv.DictReader((root / "ppa.csv").open())}
    runtime_path = root / ("task-runtime.csv" if (root / "task-runtime.csv").exists() else "runtime.csv")
    runtime = [r for r in csv.DictReader(runtime_path.open())
               if r["workload"].startswith("gemm_") and r["result_check"] == "passed"]
    breakdown = list(csv.DictReader((root / "breakdown.csv").open()))
    all_designs = [d for d in ("ieee", "hopper", "blackwell", "hopper-overlap", "blackwell-overlap",
                              "hopper-reuse-overlap", "blackwell-reuse-overlap") if d in ppa]
    designs = [d for d in all_designs if d not in ("hopper", "blackwell")] if "hopper-reuse-overlap" in ppa else all_designs
    labels = {"ieee": "IEEE FMA (B0)", "hopper": "Hopper (B1)", "blackwell": "Blackwell (B1)",
              "hopper-overlap": "Hopper dual bank", "blackwell-overlap": "Blackwell dual bank",
              "hopper-reuse-overlap": "Hopper slot reuse", "blackwell-reuse-overlap": "Blackwell slot reuse"}
    colors = {"ieee": "#666666", "hopper": "#D55E00", "blackwell": "#0072B2",
              "hopper-overlap": "#D55E00", "blackwell-overlap": "#0072B2",
              "hopper-reuse-overlap": "#D55E00", "blackwell-reuse-overlap": "#0072B2"}
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.0 if len(designs) > 3 else 2.7), layout="constrained")
    for i, d in enumerate(designs):
        parts = [float(r["area_um2"])/1000 for r in breakdown if r["design"] == d]
        axes[0].bar(i, parts[0], color=colors[d], width=.65, alpha=.6 if d.endswith("-overlap") else 1)
        axes[0].bar(i, parts[1], bottom=parts[0], color=colors[d], hatch="///", alpha=.45, width=.65)
    short = {"ieee": "IEEE", "hopper": "H B1", "blackwell": "B B1",
             "hopper-overlap": "H dual", "blackwell-overlap": "B dual",
             "hopper-reuse-overlap": "H reuse", "blackwell-reuse-overlap": "B reuse"}
    axes[0].set_xticks(range(len(designs)), [short[d] for d in designs], rotation=35)
    axes[0].set_ylabel("SKY130 cell area (10³ µm²)")
    axes[0].set_ylim(0, max(float(ppa[d]["area_um2"]) for d in designs)/1000*1.25)
    axes[0].set_title("(a) Core area")
    axes[0].legend(handles=[Patch(facecolor="#666666", label="PE + local state"),
                           Patch(facecolor="#cccccc", hatch="///", label="Mesh + forwarding")],
                   loc="upper left", frameon=False, fontsize=6)
    summary = []
    m, n = int(runtime[0]["M"]), int(runtime[0]["N"])
    for d in all_designs:
        rows = sorted((r for r in runtime if r["design"] == d), key=lambda r: int(r["K"]))
        ks = [int(r["K"]) for r in rows]
        cycles = [int(r["cycles"]) for r in rows]
        area = float(ppa[d]["area_um2"])
        style = {"ieee": dict(marker="o", linestyle="-", markersize=4),
                 "hopper": dict(marker="s", linestyle="--", markersize=6, markerfacecolor="none"),
                 "blackwell": dict(marker="^", linestyle=":", markersize=4),
                 "hopper-overlap": dict(marker="D", linestyle="-.", markersize=6, markerfacecolor="none"),
                 "blackwell-overlap": dict(marker="x", linestyle="-.", markersize=4),
                 "hopper-reuse-overlap": dict(marker="s", linestyle="--", markersize=6, markerfacecolor="none"),
                 "blackwell-reuse-overlap": dict(marker="^", linestyle=":", markersize=4)}[d]
        if d in designs:
            axes[1].plot(ks, cycles, color=colors[d], label=labels[d], **style)
            axes[2].plot(ks, [2*m*n*k/c/area*1e6 for k, c in zip(ks, cycles)], color=colors[d], **style)
        for k, c in zip(ks, cycles):
            base = next(int(r["cycles"]) for r in runtime if r["design"] == "ieee" and int(r["K"]) == k)
            summary.append({"design": d, "K": k, "area_um2": area, "cycles": c,
                            "operations_per_cycle": 2*m*n*k/c, "area_ratio_to_ieee": area/float(ppa["ieee"]["area_um2"]),
                            "cycle_ratio_to_ieee": c/base,
                            "operations_per_cycle_per_mm2": 2*m*n*k/c/area*1e6})
    axes[1].set_ylabel("Cycles / GEMM")
    axes[1].set_title("(b) Request → result")
    axes[2].set_ylabel("Operations / cycle / mm²")
    axes[2].set_title("(c) Throughput density")
    for ax in axes[1:]:
        ax.set_xlabel(f"K (M=N={m})")
        ax.set_xticks([64, 128, 256])
        ax.set_ylim(bottom=0)
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.legend(*axes[1].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(root / f"q1-area-cycles.{suffix}", dpi=300)
    plt.close(fig)
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (root / "energy.csv").open("w", newline="") as stream:
        fields = ["design", "K", "memory_scope", "status", "dynamic_power_w", "leakage_power_w", "energy_j", "method"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            writer.writerow({"design": row["design"], "K": row["K"], "memory_scope": "core-only", "status": "unmeasured"})
    print("Saved Q1 PDF/SVG/PNG and summary.csv (SKY130 core-only; no assumed frequency)")


if __name__ == "__main__":
    main(sys.argv[1])
