"""Run all 16xKx16 precision sweeps on real Qwen weight panels."""
import argparse
import hashlib
import json
from pathlib import Path

from precision_model import (configurable_dot, configurable_gdfs_dot,
                             error_metrics, os_dot, os_fp4_dot)
from qwen_corpus import (DEFAULT_TENSORS, fp32_c_matrix, make_panel,
                         quantize_e4m3, quantize_fp4_rows,
                         read_safetensors_header)
from run_precision import (COUPLING_F, FP8_F, GROUP_BITS, GROUP_F, GROUP_SIZES,
                           metrics_row, write_csv)

K_VALUES = (16, 32, 64, 128, 256)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fp8_cases(panels, k):
    cases = []
    for panel in panels:
        a = quantize_e4m3(panel["a"][:, :k])
        b = quantize_e4m3(panel["b"][:k, :])
        c = fp32_c_matrix(panel, k)
        cases.extend((a[row].tolist(), b[:, column].tolist(), c[row][column])
                     for row in range(16) for column in range(16))
    return cases


def fp4_cases(panels, k, nvfp4):
    cases = []
    for panel in panels:
        a, alpha = quantize_fp4_rows(panel["a"][:, :k], nvfp4)
        # Quantize each B column along K by viewing columns as rows.
        b_transposed, beta = quantize_fp4_rows(panel["b"][:k, :].T, nvfp4)
        c = fp32_c_matrix(panel, k)
        cases.extend((a[row].tolist(), b_transposed[column].tolist(),
                      c[row][column], alpha[row].tolist(), beta[column].tolist())
                     for row in range(16) for column in range(16))
    return cases


def common_parameters(k, panel_count, **parameters):
    return {**parameters, "K": k, "M": 16, "N": 16,
            "weight_panels": panel_count, "samples_per_point": panel_count * 256}


def run_sweeps(panels):
    fp8_rows, coupling_rows, gf_rows, gs_rows = [], [], [], []
    for k in K_VALUES:
        cases = fp8_cases(panels, k)
        blackwell = [configurable_dot(a, b, c, 25, True) for a, b, c in cases]
        os_reference = [os_dot(a, b, c) for a, b, c in cases]
        for fractional_bits in FP8_F:
            candidate = [configurable_dot(a, b, c, fractional_bits, True)
                         for a, b, c in cases]
            fp8_rows.append(metrics_row(
                "fp8_fractional_bits", "e4m3", candidate, blackwell, os_reference,
                **common_parameters(k, len(panels), F=fractional_bits, G="", GS="",
                                    coupling="c_fused")))
        for coupling in ("c_fused", "c_decoupled"):
            fused = coupling == "c_fused"
            for fractional_bits in COUPLING_F:
                candidate = [configurable_dot(a, b, c, fractional_bits, fused)
                             for a, b, c in cases]
                coupling_rows.append(metrics_row(
                    "c_coupling", "e4m3", candidate, blackwell, os_reference,
                    **common_parameters(k, len(panels), F=fractional_bits, G="", GS="",
                                        coupling=coupling)))

        for nvfp4 in (False, True):
            fmt = "nvfp4" if nvfp4 else "mxfp4"
            grouped = fp4_cases(panels, k, nvfp4)
            blackwell_fp4 = [configurable_gdfs_dot(
                a, b, c, alpha, beta, 6, 35, 16, nvfp4)
                for a, b, c, alpha, beta in grouped]
            os_fp4 = [os_fp4_dot(a, b, c, alpha, beta, nvfp4)
                      for a, b, c, alpha, beta in grouped]
            for coupling in ("c_fused", "c_decoupled"):
                fused = coupling == "c_fused"
                for fractional_bits in COUPLING_F:
                    candidate = [configurable_gdfs_dot(
                        a, b, c, alpha, beta, 6, fractional_bits, 16,
                        nvfp4, c_fused=fused)
                        for a, b, c, alpha, beta in grouped]
                    coupling_rows.append(metrics_row(
                        "c_coupling", fmt, candidate, blackwell_fp4, os_fp4,
                        **common_parameters(k, len(panels), F=fractional_bits,
                                            G=6, GS=16,
                                            coupling=coupling)))
            for group_bits in GROUP_BITS:
                for fractional_bits in GROUP_F:
                    candidate = [configurable_gdfs_dot(
                        a, b, c, alpha, beta, group_bits, fractional_bits, 16, nvfp4)
                        for a, b, c, alpha, beta in grouped]
                    gf_rows.append(metrics_row(
                        "fp4_g_f", fmt, candidate, blackwell_fp4, os_fp4,
                        **common_parameters(k, len(panels), F=fractional_bits,
                                            G=group_bits, GS=16,
                                            coupling="c_fused")))
            for group_size in GROUP_SIZES:
                candidate = [configurable_gdfs_dot(
                    a, b, c, alpha, beta, 6, 35, group_size, nvfp4)
                    for a, b, c, alpha, beta in grouped]
                gs_rows.append(metrics_row(
                    "fp4_group_size", fmt, candidate, blackwell_fp4, os_fp4,
                    **common_parameters(k, len(panels), F=35, G=6,
                                        GS=group_size, coupling="c_fused")))
    return {
        "fp8_f_sweep.csv": fp8_rows,
        "fp4_g_f_sweep.csv": gf_rows,
        "fp4_gs_sweep.csv": gs_rows,
        "c_coupling_sweep.csv": coupling_rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("qwen_results"))
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--tensor", action="append", dest="tensors",
                        help="2-D safetensors key; repeat to select several panels")
    args = parser.parse_args()
    selected = tuple(args.tensors or DEFAULT_TENSORS)
    if not args.weights.is_file():
        parser.error(f"weight file does not exist: {args.weights}")
    data_start, tensors, metadata = read_safetensors_header(args.weights)
    missing = [name for name in selected if name not in tensors]
    if missing:
        parser.error("weight tensors not found: " + ", ".join(missing))
    panels = [make_panel(args.weights, data_start, tensors[name], args.seed)
              for name in selected]
    args.output.mkdir(parents=True, exist_ok=True)
    files = run_sweeps(panels)
    all_rows = []
    for name, rows in files.items():
        write_csv(args.output / name, rows)
        all_rows.extend(rows)
    write_csv(args.output / "all_precision_sweeps.csv", all_rows)
    output_names = (*files, "all_precision_sweeps.csv")
    sources = ("precision_model.py", "qwen_corpus.py", "run_precision.py",
               "run_qwen_precision.py")
    manifest = {
        "base_commit": "f197fdfc52118d654740ac747134e207bf0483e7",
        "weights": {
            "file": args.weights.name, "bytes": args.weights.stat().st_size,
            "sha256": sha256_file(args.weights), "metadata": metadata,
        },
        "seed": args.seed, "M": 16, "N": 16, "K": list(K_VALUES),
        "systolic_array": "16x16",
        "panels": [{key: panel[key] for key in
                    ("name", "dtype", "shape", "row_start", "column_start")}
                   for panel in panels],
        "activation_policy": "seeded N(0,1), clipped to [-6,6]; shared 16x256 panel prefixes",
        "c_policy": "seeded signed N(0,1) scaled by sqrt(K)*sample_std(weight panel)",
        "fp8_quantization": "nearest finite E4M3; numeric ties choose lower value",
        "mxfp4_quantization": "E2M1 with per-row/per-column 32-value E8M0 ceil scale",
        "nvfp4_quantization": "E2M1 with per-row/per-column 16-value UE4M3 ceil scale",
        "nvfp4_tensor_scale": "fixed to 1.0; the external FP32 tensor scale is outside MMA",
        "short_fp4_K": "K<64 is zero-padded to one Blackwell 64-lane GDFS instruction",
        "relative_error": "signed MRE uses reference; MARE and directional signed error use abs(reference); zero references excluded and counted",
        "blackwell_reference": "FP8 F=25; FP4 G=6,F=35,GS=16,C-fused",
        "os_baseline": "exact low-precision product followed by sequential IEEE FP32 RNE FMA",
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in sources
        },
        "files": {name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
                  for name in output_names},
        "rows": len(all_rows),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"rows": len(all_rows), "samples_per_point": len(panels) * 256,
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
