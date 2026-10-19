"""Sequential CPU differential checks for dedicated Blackwell MXFP4/NVFP4."""
import argparse
import importlib.metadata
from pathlib import Path
import random

from grouped_reference import gst_fdpa, gst_dot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=1000, help="cases per FP4 format")
    parser.add_argument("--seed", type=int, default=2899)
    parser.add_argument("--vectors", type=Path)
    parser.add_argument("--matrix-vectors", type=Path)
    args = parser.parse_args()
    assert args.cases > 0
    import torch
    from mmasim.arithmetic.fdpa import MMA_GST_FDPA
    from mmasim.arithmetic.helper import fp4_value_table
    assert importlib.metadata.version("mmasim") == "2.0.3"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    vectors = ["# kind a_hex_csv b_hex_csv c_hex alpha_hex_csv beta_hex_csv d_hex"]
    matrices = ["# kind k a_row_major b_row_major c_row_major alpha_row_major beta_row_major d_row_major"]
    csv = lambda xs: ",".join(f"{x:x}" for x in xs)
    flatten = lambda matrix: [x for row in matrix for x in row]
    for nvfp4 in (False, True):
        kind = "nvfp4" if nvfp4 else "mxfp4"
        rng = random.Random(f"{args.seed}:{nvfp4}")
        scales = 4 if nvfp4 else 2
        block = 16 if nvfp4 else 32
        neutral = 0x38 if nvfp4 else 127
        oracle = MMA_GST_FDPA(16, 35, "RZ-FP32", 64, -139)
        def scale_tensor(raw):
            bits = torch.tensor(raw, dtype=torch.uint8)
            if nvfp4:
                bits = bits & 127  # UE4M3 conversion in the author's instruction wrapper
            return bits.view(torch.float8_e4m3fn if nvfp4 else torch.float8_e8m0fnu)
        def verify(aa, bb, cc, alphas, betas):
            a = fp4_value_table[torch.tensor(aa)]
            b = fp4_value_table[torch.tensor(bb)]
            c = torch.tensor(cc, dtype=torch.uint32).view(torch.float32)
            got = oracle.dpa(a, b, c, scale_tensor(alphas), scale_tensor(betas)).view(torch.uint32).tolist()
            for i in range(len(aa)):
                expected = gst_fdpa(aa[i], bb[i], cc[i], alphas[i], betas[i], nvfp4)
                assert expected == got[i], dict(nvfp4=nvfp4, a=aa[i], b=bb[i],
                    c=hex(cc[i]), alpha=alphas[i], beta=betas[i],
                    integer=hex(expected), mmasim=hex(got[i]))
            return got
        def record(av, bv, cv, alpha, beta):
            result = verify([av], [bv], [cv], [alpha], [beta])[0]
            vectors.append(" ".join((kind, csv(av), csv(bv), f"{cv:x}", csv(alpha), csv(beta), f"{result:x}")))
        if args.vectors:
            for av, bv, cv in [([0]*64, [0]*64, 0), ([0]*64, [0]*64, 1),
                               ([2]*64, [2]*64, 0), ([7]*64, [7]*64, 0),
                               ([7]*64, [15]*64, 0), ([2]*64, [2]*64, 0x7f800000),
                               ([2]*64, [2]*64, 0xff800000), ([2]*64, [2]*64, 0x7fa12345)]:
                record(av, bv, cv, [neutral]*scales, [neutral]*scales)
            high = 0x7e if nvfp4 else 200
            record([2]*8+[10]*8+[0]*48, [2]*16+[0]*48, 0x33000000,
                   [high]+[neutral]*(scales-1), [high]+[neutral]*(scales-1))
            record([0]*64, [0]*64, 0x33000000, [high]*scales, [high]*scales)
            # All scale encodings, including NVFP4's ignored high storage bit.
            for code in range(256):
                record([7]*64, [7]*64, 0, [code]*scales, [neutral]*scales)
            # Exhaust every FP4 multiplication encoding without cancellation.
            if not nvfp4:
                for ai in range(16):
                    for bi in range(16):
                        record([ai]+[0]*63, [bi]+[0]*63, 0, [neutral]*scales, [neutral]*scales)
        count = 0
        while count < args.cases:
            n = min(128, args.cases - count)
            aa = [[rng.randrange(16) for _ in range(64)] for _ in range(n)]
            bb = [[rng.randrange(16) for _ in range(64)] for _ in range(n)]
            cc = [rng.getrandbits(32) for _ in range(n)]
            alphas = [[rng.randrange(256) for _ in range(scales)] for _ in range(n)]
            betas = [[rng.randrange(256) for _ in range(scales)] for _ in range(n)]
            got = verify(aa, bb, cc, alphas, betas)
            if args.vectors:
                for i in range(min(n, max(0, 48-count))):
                    vectors.append(" ".join((kind, csv(aa[i]), csv(bb[i]), f"{cc[i]:x}",
                                              csv(alphas[i]), csv(betas[i]), f"{got[i]:x}")))
            count += n
        print(f"PASS {'NVFP4' if nvfp4 else 'MXFP4'}: {count} cases", flush=True)
        if args.matrix_vectors:
            mrng = random.Random(f"matrix:{args.seed}:{nvfp4}")
            for k in (64, 128, 192):
                for raw in (False, True):
                    a = [[mrng.randrange(16) for _ in range(k)] for _ in range(2)]
                    b = [[mrng.randrange(16) for _ in range(2)] for _ in range(k)]
                    c = [[mrng.getrandbits(32) if raw else [0x3f800000, 0xbf000000,
                         0x33800000, 1][r*2+col] for col in range(2)] for r in range(2)]
                    def scale_code():
                        return mrng.randrange(256) if raw else mrng.choice(
                            [0, 0x28, 0x38, 0x40, 0x44, 0x58] if nvfp4 else list(range(123, 132)))
                    alpha = [[scale_code() for _ in range(k//block)] for _ in range(2)]
                    beta = [[scale_code() for _ in range(2)] for _ in range(k//block)]
                    author = oracle(fp4_value_table[torch.tensor(a)], fp4_value_table[torch.tensor(b)],
                                    torch.tensor(c, dtype=torch.uint32).view(torch.float32),
                                    scale_tensor(alpha), scale_tensor(beta)).view(torch.uint32).tolist()
                    expected = [[gst_dot(a[r], [b[t][col] for t in range(k)], c[r][col], alpha[r],
                        [beta[t][col] for t in range(k//block)], nvfp4) for col in range(2)] for r in range(2)]
                    assert author == expected, (kind, k, author, expected)
                    matrices.append(" ".join((kind, str(k), csv(flatten(a)), csv(flatten(b)),
                        csv(flatten(c)), csv(flatten(alpha)), csv(flatten(beta)), csv(flatten(author)))))
    for path, lines in ((args.vectors, vectors), (args.matrix_vectors, matrices)):
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(lines)+"\n")
            print(f"Exported {len(lines)-1} vectors to {path}")


if __name__ == "__main__":
    main()
