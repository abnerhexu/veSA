"""Sequential CPU differential validation and reproducible RTL vector export.

Run from the Gemmini submodule:
  ../.venv/bin/python modeling/tensorcore/validate.py --cases 10000
MMA-Sim 2.0.3 / CPU torch 2.9.0 are optional validation dependencies; the
reference.py model itself uses the Python standard library only.
"""
import argparse
import importlib.metadata
from pathlib import Path
import random

from reference import FORMATS, fdpa, dot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=1000, help="cases per supported format/output")
    parser.add_argument("--seed", type=int, default=776)
    parser.add_argument("--architecture", choices=("blackwell", "hopper"), default="blackwell")
    parser.add_argument("--vectors", type=Path, help="write a bounded subset of verified RTL vectors")
    parser.add_argument("--matrix-vectors", type=Path, help="write author-verified 2x2 matrix vectors")
    args = parser.parse_args()
    assert args.cases > 0
    import torch
    from mmasim.arithmetic.fdpa import MMA_T_FDPA
    assert importlib.metadata.version("mmasim") == "2.0.3", "Reference version must stay pinned"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    dtypes = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2,
              "fp16": torch.float16, "bf16": torch.bfloat16}
    matrix_rng = random.Random(args.seed + 2899)
    vectors = ["# format output a_hex_csv b_hex_csv c_hex d_hex; seed=" + str(args.seed)]
    matrices = ["# format output k a_row_major b_row_major c_row_major d_row_major"]
    total = 0
    for name, fmt in FORMATS.items():
        for half in (False, True):
            if name == "bf16" and half:
                continue
            rng = random.Random(f"{args.seed}:{name}:{half}")
            length = 32 if fmt.width == 8 else 16
            hopper = args.architecture == "hopper"
            fractional_bits = 13 if hopper and fmt.width == 8 else 25
            rho = "RNE-FP16" if half else "RZ-E8M13" if hopper and fmt.width == 8 else "RZ-FP32"
            oracle = MMA_T_FDPA(fractional_bits, rho, length,
                               -22 if half and not hopper else -133)
            int_dtype = torch.uint8 if fmt.width == 8 else torch.uint16
            def verified_vector(av, bv, cv):
                a = torch.tensor([av], dtype=int_dtype).view(dtypes[name])
                b = torch.tensor([bv], dtype=int_dtype).view(dtypes[name])
                c = torch.tensor([cv], dtype=torch.uint16 if half else torch.uint32).view(
                    torch.float16 if half else torch.float32)
                got = oracle.dpa(a, b, c).view(torch.uint16 if half else torch.uint32).item()
                assert got == fdpa(av, bv, cv, name, half, args.architecture), (name, half, av, bv, cv, got)
                vectors.append(" ".join((name, "f16" if half else "f32",
                    ",".join(f"{x:x}" for x in av), ",".join(f"{x:x}" for x in bv),
                    f"{cv:x}", f"{got:x}")))
            one = ((1 << (fmt.exponent - 1)) - 1) << fmt.fraction
            sign = 1 << (fmt.width - 1)
            # Short lengths, signed zero, subnormals, exact cancellation and C-only.
            for av, bv, cv in [([0], [0], 0), ([0], [0], 1), ([1], [1], 0),
                               ([one], [one], 0), ([one, one | sign], [one, one], 0),
                               ([0], [0], 0x8000 if half else 0x80000000)]:
                verified_vector(av, bv, cv)
            if hopper:
                # C-only FP32 rounding and negative RZ, including reduced
                # precision subnormals. Blackwell fixture generation is unchanged.
                for cv in ((1, 0x3ff, 0x400, 0x7fffff, 0x800000, 0x3f8007ff,
                            0xbf8007ff, 0x7f7fffff, 0xff7fffff) if not half else (1, 0x8001)):
                    verified_vector([0], [0], cv)
                if name == "fp16" and half:
                    # Half-subnormal rounding tie plus an extremely small term.
                    verified_vector([0x0800, 1], [0x0c00, 1], 0)
                if fmt.width == 8 and not half:
                    # Growth to exponent 4 makes block-end E8M13 rounding
                    # observable separately from F=13 alignment truncation.
                    tiny_a, tiny_b = (0x08, 0x04) if name == "e4m3" else (0x08, 0x3c)
                    verified_vector([one]*31 + [tiny_a], [one]*31 + [tiny_b], 0)
                    verified_vector([one | sign]*31 + [tiny_a | sign], [one]*31 + [tiny_b], 0)
            if not half:
                one_and_half = one | (1 << (fmt.fraction - 1))
                verified_vector([one_and_half, one_and_half | sign],
                                [one_and_half, one_and_half], 0x33000000)
                if name == "fp16":
                    verified_vector([0xf000, 0xb800, 0xb400, 0xb000],
                                    [0x6400, 0x3c00, 0x3c00, 0x3c00], 0x4b000000)
            count = 0
            # Bounded batches keep memory small even for a million-case run.
            while count < args.cases:
                n = min(256, args.cases - count)
                aa = [[rng.getrandbits(fmt.width) for _ in range(length)] for _ in range(n)]
                bb = [[rng.getrandbits(fmt.width) for _ in range(length)] for _ in range(n)]
                cc = [rng.getrandbits(16 if half else 32) for _ in range(n)]
                # Mix zeros and sparse adversarial inputs into raw-bit sampling.
                for i in range(n):
                    if (count + i) % 4 == 0:
                        aa[i][4:] = [0] * (length - 4)
                        bb[i][4:] = [0] * (length - 4)
                        cc[i] = 0
                a = torch.tensor(aa, dtype=int_dtype).view(dtypes[name])
                b = torch.tensor(bb, dtype=int_dtype).view(dtypes[name])
                c = torch.tensor(cc, dtype=torch.uint16 if half else torch.uint32).view(
                    torch.float16 if half else torch.float32)
                actual = oracle.dpa(a, b, c).view(torch.uint16 if half else torch.uint32).tolist()
                for i, (av, bv, cv, got) in enumerate(zip(aa, bb, cc, actual)):
                    expected = fdpa(av, bv, cv, name, half, args.architecture)
                    if got != expected:
                        raise AssertionError(dict(format=name, half=half, case=count + i,
                                                  a=av, b=bv, c=hex(cv),
                                                  integer=hex(expected), mmasim=hex(got)))
                    if count + i < 48:
                        vectors.append(" ".join((name, "f16" if half else "f32",
                            ",".join(f"{x:x}" for x in av), ",".join(f"{x:x}" for x in bv),
                            f"{cv:x}", f"{got:x}")))
                count += n
            total += count
            print(f"PASS {name}-{'f16' if half else 'f32'}: {count} raw-bit cases", flush=True)
            if args.matrix_vectors:
                for k in (1, 17, 32, 33, 64, 65):
                    for raw in (False, True):
                        def element():
                            value = matrix_rng.getrandbits(fmt.width)
                            if not raw:
                                # Finite moderate exponents, signs and significands vary.
                                value = (value & ((1 << fmt.fraction) - 1)) | (
                                    matrix_rng.randrange((1 << (fmt.exponent - 1)) - 3,
                                                         (1 << (fmt.exponent - 1)) + 1) << fmt.fraction) | (
                                    matrix_rng.randrange(2) << (fmt.width - 1))
                            return value
                        av = [[element() for _ in range(k)] for _ in range(2)]
                        bv = [[element() for _ in range(2)] for _ in range(k)]
                        cv = [[matrix_rng.getrandbits(16 if half else 32) if raw else 0
                               for _ in range(2)] for _ in range(2)]
                        a = torch.tensor(av, dtype=int_dtype).view(dtypes[name])
                        b = torch.tensor(bv, dtype=int_dtype).view(dtypes[name])
                        c = torch.tensor(cv, dtype=torch.uint16 if half else torch.uint32).view(
                            torch.float16 if half else torch.float32)
                        got = oracle(a, b, c).view(torch.uint16 if half else torch.uint32).tolist()
                        expected = [[dot(av[r], [bv[t][col] for t in range(k)], cv[r][col],
                                         name, half, args.architecture) for col in range(2)] for r in range(2)]
                        assert got == expected, (name, half, k, got, expected)
                        csv = lambda matrix: ",".join(f"{x:x}" for row in matrix for x in row)
                        matrices.append(" ".join((name, "f16" if half else "f32", str(k),
                                                  csv(av), csv(bv), csv(cv), csv(got))))
                if hopper and fmt.width == 8:
                    # A 33rd term cancels the integer part after the first
                    # block has rounded. Rounding only once at K end is wrong.
                    tiny_a, tiny_b = (0x08, 0x04) if name == "e4m3" else (0x08, 0x3c)
                    thirty_two = (((1 << (fmt.exponent - 1)) - 1) + 5) << fmt.fraction
                    av = [[one]*31 + [tiny_a, one | sign] for _ in range(2)]
                    bv = [[one]*2 for _ in range(31)] + [[tiny_b]*2, [thirty_two]*2]
                    cv = [[0, 0], [0, 0]]
                    a = torch.tensor(av, dtype=int_dtype).view(dtypes[name])
                    b = torch.tensor(bv, dtype=int_dtype).view(dtypes[name])
                    c = torch.tensor(cv, dtype=torch.uint16 if half else torch.uint32).view(
                        torch.float16 if half else torch.float32)
                    got = oracle(a, b, c).view(torch.uint16 if half else torch.uint32).tolist()
                    expected = [[dot(av[r], [bv[t][col] for t in range(33)], 0,
                                     name, half, args.architecture) for col in range(2)] for r in range(2)]
                    assert got == expected, (name, half, got, expected)
                    csv = lambda matrix: ",".join(f"{x:x}" for row in matrix for x in row)
                    matrices.append(" ".join((name, "f16" if half else "f32", "33",
                                              csv(av), csv(bv), csv(cv), csv(got))))
    if args.vectors:
        args.vectors.parent.mkdir(parents=True, exist_ok=True)
        args.vectors.write_text("\n".join(vectors) + "\n")
        print(f"Exported {len(vectors)-1} vectors to {args.vectors}")
    if args.matrix_vectors:
        args.matrix_vectors.parent.mkdir(parents=True, exist_ok=True)
        args.matrix_vectors.write_text("\n".join(matrices) + "\n")
        print(f"Exported {len(matrices)-1} matrices to {args.matrix_vectors}")
    print(f"PASS total={total}; architecture={args.architecture}; mmasim=2.0.3; torch={torch.__version__}; seed={args.seed}")


if __name__ == "__main__":
    main()
