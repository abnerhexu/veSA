"""Small Q1 corpus: fixed release cases plus K=64/128/256 throughput tasks."""
import ctypes
import argparse
import math
from pathlib import Path
import random
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "modeling/tensorcore"))
from reference import decode, FORMATS, dot


def f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def bits32(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def e4m3(bits):
    if bits & 127 == 0:
        return -0.0 if bits & 128 else 0.0
    sig, exp, frac, special = decode(bits, FORMATS["e4m3"])
    return math.nan if special else math.ldexp(sig, exp-frac)


def tiled_cases(dim, problem_dim):
    """Identical global tasks at each array size; only output tiling changes."""
    rng = random.Random(2899)
    finite = [x for x in range(256) if x & 127 != 127]
    cases = []
    for k in (1, 31, 32, 33, 65, 64, 128, 256, 512):
        a = rng.choices(finite, k=problem_dim*k)
        b = rng.choices(finite, k=problem_dim*k)
        c = [([0x3f800000, 0xbf800000, 0x35800000, 0])[i % 4] for i in range(problem_dim**2)]
        cases.append((f"gemm_k{k}" if k in (32, 64, 128, 256, 512) else f"tail_k{k}", k, a, b, c))
    cases.append(("special", 1, ([0x80, 0, 0x7f, 0x7e] * problem_dim)[:problem_dim],
                  ([0x38, 0xb8, 1, 0xfe] * problem_dim)[:problem_dim], [0x80000000]*problem_dim**2))
    tiled = []
    for name, k, a, b, c in cases:
        for row in range(0, problem_dim, dim):
            for col in range(0, problem_dim, dim):
                tiled.append((f"{name}_tile{row}_{col}", k,
                              a[row*k:(row+dim)*k],
                              [b[t*problem_dim+j] for t in range(k) for j in range(col, col+dim)],
                              [c[r*problem_dim+j] for r in range(row, row+dim) for j in range(col, col+dim)]))
    return tiled


def retile(source, path, dim, problem_dim):
    output = [f"# Retiled author-checked global corpus; array_dim={dim}; problem_dim={problem_dim}; source={source}"]
    count = 0
    for line in Path(source).read_text().splitlines():
        if line.startswith("#"):
            continue
        v = line.split()
        name, k = v[0].removesuffix("_tile0_0"), int(v[1])
        arrays = [[int(x, 16) for x in field.split(",")] for field in v[2:]]
        assert v[0].endswith("_tile0_0") and len(arrays[2]) == problem_dim**2
        for row in range(0, problem_dim, dim):
            for col in range(0, problem_dim, dim):
                tile = [arrays[0][row*k:(row+dim)*k],
                        [arrays[1][t*problem_dim+j] for t in range(k) for j in range(col, col+dim)]]
                for field in arrays[2:]:
                    tile.append([field[r*problem_dim+j] for r in range(row, row+dim) for j in range(col, col+dim)])
                output.append(" ".join([f"{name}_tile{row}_{col}", str(k)] + [",".join(f"{x:x}" for x in field) for field in tile]))
                count += 1
    Path(path).write_text("\n".join(output) + "\n")
    print(f"Retiled {count} matrices from the checked global corpus; no repeated author-model execution")


def generate(path, dim=2, problem_dim=None, source=None):
    if source is not None:
        return retile(source, path, dim, problem_dim)
    import torch
    from mmasim.arithmetic.fdpa import MMA_T_FDPA
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    fma = ctypes.CDLL("libm.so.6").fmaf
    fma.argtypes = [ctypes.c_float] * 3
    fma.restype = ctypes.c_float
    cases = []
    for line in (ROOT / "modeling/tensorcore/matrices.tsv").read_text().splitlines():
        if line.startswith("#"):
            continue
        v = line.split()
        if v[:2] == ["e4m3", "f32"]:
            arrays = [[int(x, 16) for x in v[i].split(",")] for i in (3, 4, 5)]
            cases.append((f"release_{len(cases)}", int(v[2]), *arrays))
    for i, aa in enumerate(([0, 1], [2, 3], [4, 5], [6, 7],
                            [0x7e, 0xfe], [0x7f, 0xff], [0x80, 0x81])):
        cases.append((f"encoding_{i}", 1, aa, [0x38, 0xb8], [0, 0, 0, 0]))
    cases.append(("signed_zero", 1, [0x80, 0], [0x38, 0xb8], [0x80000000]*4))
    rng = random.Random(2899)
    for k in (64, 128, 256):
        # Finite raw encodings span exponent, sign, subnormal and cancellation cases.
        finite = [x for x in range(256) if x & 127 != 127]
        a = rng.choices(finite, k=2*k)
        b = rng.choices(finite, k=2*k)
        cases.append((f"gemm_k{k}", k, a, b, [0x3f800000, 0xbf800000, 0x35800000, 0]))
    if problem_dim is not None:
        cases = tiled_cases(dim, problem_dim)
    oracles = {arch: MMA_T_FDPA(f, rho, 32, -133) for arch, f, rho in
               [("blackwell", 25, "RZ-FP32"), ("hopper", 13, "RZ-E8M13")]}
    output = [f"# name k A_row_major B_row_major C IEEE_RNE Blackwell Hopper; seed=2899; array_dim={dim}; problem_dim={problem_dim or dim}"]
    for name, k, a, b, c in cases:
        results = {key: [] for key in ["ieee", "blackwell", "hopper"]}
        for r in range(dim):
            for col in range(dim):
                av, bv, cv = a[r*k:(r+1)*k], b[col::dim], c[dim*r+col]
                acc = f32(cv)
                for ai, bi in zip(av, bv):
                    acc = fma(e4m3(ai), e4m3(bi), acc)
                results["ieee"].append(0x7fc00000 if math.isnan(acc) else bits32(acc))
                for arch, oracle in oracles.items():
                    expected = dot(av, bv, cv, architecture=arch)
                    actual = cv
                    for start in range(0, k, 32):
                        ta = torch.tensor([av[start:start+32]], dtype=torch.uint8).view(torch.float8_e4m3fn)
                        tb = torch.tensor([bv[start:start+32]], dtype=torch.uint8).view(torch.float8_e4m3fn)
                        tc = torch.tensor([actual], dtype=torch.uint32).view(torch.float32)
                        actual = oracle.dpa(ta, tb, tc).view(torch.uint32).item()
                    assert actual == expected, (name, arch, r, col, actual, expected)
                    results[arch].append(expected)
        arrays = [a, b, c, results["ieee"], results["blackwell"], results["hopper"]]
        output.append(" ".join([name, str(k)] + [",".join(f"{x:x}" for x in v) for v in arrays]))
    Path(path).write_text("\n".join(output) + "\n")
    print(f"Author-checked {len(cases)} matrices for both profiles; IEEE oracle: libc fmaf RNE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("dim", type=int, nargs="?", default=2)
    parser.add_argument("problem_dim", type=int, nargs="?")
    parser.add_argument("--source", type=Path, help="Retile a checked corpus containing one full-size tile per global task")
    args = parser.parse_args()
    generate(args.output, args.dim, args.problem_dim, args.source)
