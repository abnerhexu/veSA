"""Run deterministic 16x16 precision sweeps and write reviewable CSV files."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import struct

from precision_model import (configurable_dot, configurable_gdfs, error_metrics,
                             fp32_bits, os_dot, os_fp4_dot)

FP8_F = (13, 16, 19, 22, 25)
GROUP_BITS = (3, 4, 5, 6, 7, 8)
GROUP_F = (26, 29, 32, 35)
GROUP_SIZES = (12, 14, 16, 18, 20)
COUPLING_F = (6, 7, 8, 9, 10)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def e4m3(rng):
    if rng.random() < 0.04:
        return rng.choice((0x00, 0x80))
    sign = rng.randrange(2) << 7
    exponent = rng.randrange(2, 14) << 3
    return sign | exponent | rng.randrange(8)


def fp4(rng):
    if rng.random() < 0.05:
        return rng.choice((0, 8))
    return (rng.randrange(2) << 3) | rng.randrange(1, 8)


def c_value(rng):
    sign = -1.0 if rng.randrange(2) else 1.0
    return fp32_bits(sign * (0.5 + rng.random()) * (2.0 ** rng.randrange(-5, 6)))


def scale_code(rng, nvfp4):
    if nvfp4:
        return (rng.randrange(3, 12) << 3) | rng.randrange(8)
    return rng.randrange(121, 134)


def fp8_cases(trials, m, n, k, seed):
    rng = random.Random(f"fp8:{seed}")
    cases = []
    for _ in range(trials):
        a = [[e4m3(rng) for _ in range(k)] for _ in range(m)]
        b = [[e4m3(rng) for _ in range(n)] for _ in range(k)]
        c = [[c_value(rng) for _ in range(n)] for _ in range(m)]
        cases.extend((a[row], [b[index][column] for index in range(k)], c[row][column])
                     for row in range(m) for column in range(n))
    return cases


def fp4_cases(trials, m, n, k, seed, nvfp4):
    rng = random.Random(f"fp4:{seed}:{nvfp4}")
    block = 16 if nvfp4 else 32
    cases = []
    for _ in range(trials):
        a = [[fp4(rng) for _ in range(k)] for _ in range(m)]
        b = [[fp4(rng) for _ in range(n)] for _ in range(k)]
        c = [[c_value(rng) for _ in range(n)] for _ in range(m)]
        alpha = [[scale_code(rng, nvfp4) for _ in range(k // block)] for _ in range(m)]
        beta = [[scale_code(rng, nvfp4) for _ in range(n)] for _ in range(k // block)]
        cases.extend((a[row], [b[index][column] for index in range(k)], c[row][column],
                      alpha[row], [beta[index][column] for index in range(k // block)])
                     for row in range(m) for column in range(n))
    return cases


def metrics_row(experiment, fmt, candidate, blackwell, os_reference, **parameters):
    row = {"experiment": experiment, "format": fmt, **parameters,
           "samples": len(candidate)}
    row.update(error_metrics(candidate, blackwell, "vs_blackwell"))
    row.update(error_metrics(candidate, os_reference, "vs_os"))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("results"))
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    assert args.trials > 0 and args.m > 0 and args.n > 0
    args.output.mkdir(parents=True, exist_ok=True)

    fp8 = fp8_cases(args.trials, args.m, args.n, 256, args.seed)
    fp8_blackwell = [configurable_dot(a, b, c, 25, True) for a, b, c in fp8]
    fp8_os = [os_dot(a, b, c) for a, b, c in fp8]
    fp8_rows = []
    for fractional_bits in FP8_F:
        candidate = [configurable_dot(a, b, c, fractional_bits, True)
                     for a, b, c in fp8]
        fp8_rows.append(metrics_row("fp8_fractional_bits", "e4m3", candidate,
                                    fp8_blackwell, fp8_os, F=fractional_bits,
                                    G="", GS="", coupling="c_fused", K=256,
                                    M=args.m, N=args.n, trials=args.trials))

    coupling_rows = []
    for coupling in ("c_fused", "c_decoupled"):
        fused = coupling == "c_fused"
        for fractional_bits in COUPLING_F:
            candidate = [configurable_dot(a, b, c, fractional_bits, fused)
                         for a, b, c in fp8]
            coupling_rows.append(metrics_row("c_coupling", "e4m3", candidate,
                                              fp8_blackwell, fp8_os,
                                              F=fractional_bits, G="", GS="",
                                              coupling=coupling, K=256,
                                              M=args.m, N=args.n,
                                              trials=args.trials))

    gf_rows, gs_rows = [], []
    for nvfp4 in (False, True):
        fmt = "nvfp4" if nvfp4 else "mxfp4"
        cases = fp4_cases(args.trials, args.m, args.n, 64, args.seed, nvfp4)
        blackwell = [configurable_gdfs(a, b, c, alpha, beta, 6, 35, 16, nvfp4)
                     for a, b, c, alpha, beta in cases]
        os_reference = [os_fp4_dot(a, b, c, alpha, beta, nvfp4)
                        for a, b, c, alpha, beta in cases]
        for group_bits in GROUP_BITS:
            for fractional_bits in GROUP_F:
                candidate = [configurable_gdfs(a, b, c, alpha, beta,
                                               group_bits, fractional_bits, 16, nvfp4)
                             for a, b, c, alpha, beta in cases]
                gf_rows.append(metrics_row("fp4_g_f", fmt, candidate, blackwell,
                                           os_reference, F=fractional_bits,
                                           G=group_bits, GS=16,
                                           coupling="c_fused", K=64,
                                           M=args.m, N=args.n,
                                           trials=args.trials))
        for group_size in GROUP_SIZES:
            candidate = [configurable_gdfs(a, b, c, alpha, beta,
                                           6, 35, group_size, nvfp4)
                         for a, b, c, alpha, beta in cases]
            gs_rows.append(metrics_row("fp4_group_size", fmt, candidate,
                                       blackwell, os_reference, F=35, G=6,
                                       GS=group_size, coupling="c_fused", K=64,
                                       M=args.m, N=args.n,
                                       trials=args.trials))

    files = {
        "fp8_f_sweep.csv": fp8_rows,
        "fp4_g_f_sweep.csv": gf_rows,
        "fp4_gs_sweep.csv": gs_rows,
        "c_coupling_sweep.csv": coupling_rows,
    }
    all_rows = []
    for name, rows in files.items():
        write_csv(args.output / name, rows)
        all_rows.extend(rows)
    write_csv(args.output / "all_precision_sweeps.csv", all_rows)
    manifest = {
        "base_commit": "f197fdfc52118d654740ac747134e207bf0483e7",
        "seed": args.seed, "trials": args.trials, "M": args.m, "N": args.n,
        "fp8_K": 256, "fp4_K": 64,
        "input_policy": "finite signed raw encodings with exponent spread; signed nonzero FP32 C",
        "relative_error": "signed MRE uses reference; MARE and directional signed error use abs(reference); zero references excluded and counted",
        "os_baseline": "exact low-precision product followed by sequential IEEE FP32 RNE FMA",
        "source_sha256": {
            name: hashlib.sha256((Path(__file__).with_name(name)).read_bytes()).hexdigest()
            for name in ("precision_model.py", "run_precision.py")
        },
        "files": {name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
                  for name in (*files, "all_precision_sweeps.csv")},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"rows": len(all_rows), "output": str(args.output),
                      "files": list(manifest["files"])}, indent=2))


if __name__ == "__main__":
    main()
