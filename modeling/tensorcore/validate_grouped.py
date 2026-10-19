"""Sequential CPU differential checks for dedicated Blackwell MXFP4/NVFP4."""
import argparse
import importlib.metadata
import random

from grouped_reference import gst_fdpa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=1000, help="cases per FP4 format")
    parser.add_argument("--seed", type=int, default=2899)
    args = parser.parse_args()
    assert args.cases > 0
    import torch
    from mmasim.arithmetic.fdpa import gst_fdpa as author_gst
    from mmasim.arithmetic.helper import fp4_value_table
    assert importlib.metadata.version("mmasim") == "2.0.3"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for nvfp4 in (False, True):
        rng = random.Random(f"{args.seed}:{nvfp4}")
        scales = 4 if nvfp4 else 2
        count = 0
        while count < args.cases:
            n = min(128, args.cases - count)
            aa = [[rng.randrange(16) for _ in range(64)] for _ in range(n)]
            bb = [[rng.randrange(16) for _ in range(64)] for _ in range(n)]
            cc = [rng.getrandbits(32) for _ in range(n)]
            alphas = [[rng.randrange(128 if nvfp4 else 256) for _ in range(scales)] for _ in range(n)]
            betas = [[rng.randrange(128 if nvfp4 else 256) for _ in range(scales)] for _ in range(n)]
            a = fp4_value_table[torch.tensor(aa)]
            b = fp4_value_table[torch.tensor(bb)]
            c = torch.tensor(cc, dtype=torch.uint32).view(torch.float32)
            scale_dtype = torch.float8_e4m3fn if nvfp4 else torch.float8_e8m0fnu
            alpha = torch.tensor(alphas, dtype=torch.uint8).view(scale_dtype)
            beta = torch.tensor(betas, dtype=torch.uint8).view(scale_dtype)
            got = author_gst(a, b, c, alpha, beta, 16, 35, "RZ-FP32", -139).view(torch.uint32).tolist()
            for i in range(n):
                expected = gst_fdpa(aa[i], bb[i], cc[i], alphas[i], betas[i], nvfp4)
                assert expected == got[i], dict(nvfp4=nvfp4, index=count + i, a=aa[i], b=bb[i],
                    c=hex(cc[i]), alpha=alphas[i], beta=betas[i],
                    integer=hex(expected), mmasim=hex(got[i]))
            count += n
        print(f"PASS {'NVFP4' if nvfp4 else 'MXFP4'}: {count} cases", flush=True)


if __name__ == "__main__":
    main()
