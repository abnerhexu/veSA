"""Integer oracle for paper776 Algorithm 9, dedicated Blackwell FP4 path.

Scale blocks: 32 for MXFP4, 16 for NVFP4. Computation groups: always 16.
The result of a group that cancels to zero retains its scale exponent unless
all original products were zero, following MMA-Sim 2.0.3 gst_fdpa.
"""
from reference import FP32, FORMATS, decode, pack, shift_rz

HALF_UNITS = (0, 1, 2, 3, 4, 6, 8, 12)


def fp4_half_units(bits):
    assert 0 <= bits < 16
    magnitude = HALF_UNITS[bits & 7]
    return -magnitude if bits & 8 else magnitude


def scale(bits, nvfp4):
    assert 0 <= bits < 256
    if nvfp4:
        return decode(bits & 127, FORMATS["e4m3"])
    return 1, bits - 127, 0, "nan" if bits == 255 else None


def gst_fdpa(a, b, c, alpha, beta, nvfp4=False):
    assert len(a) == len(b) == 64, "Dedicated FP4 path requires a 64-element block"
    block = 16 if nvfp4 else 32
    assert len(alpha) == len(beta) == 64 // block
    terms = []
    nan = False
    for group in range(4):
        products = [fp4_half_units(a[k]) * fp4_half_units(b[k])
                    for k in range(group * 16, (group + 1) * 16)]
        p = sum(products)  # quarter units, exact
        index = group * 16 // block
        sa, ea, fa, special_a = scale(alpha[index], nvfp4)
        sb, eb, fb, special_b = scale(beta[index], nvfp4)
        nan |= special_a == "nan" or special_b == "nan"
        exponent = -139 if not any(products) or not sa or not sb else ea + eb
        terms.append((p * sa * sb, exponent, 2 + fa + fb))
    sc, ec, fc, special_c = decode(c, FP32)
    if nan or special_c == "nan":
        return 0x7fffffff
    if special_c == "inf":
        return 0xff800000 if sc < 0 else 0x7f800000
    terms.append((sc, ec if sc else -139, fc))
    maximum = max(e for _, e, _ in terms)
    total = sum(shift_rz(s, 35 - f + e - maximum) for s, e, f in terms)
    return pack(total, maximum - 35)


def gst_dot(a, b, c, alpha, beta, nvfp4=False):
    assert len(a) == len(b) and len(a) > 0 and len(a) % 64 == 0
    block = 16 if nvfp4 else 32
    assert len(alpha) == len(beta) == len(a) // block
    for start in range(0, len(a), 64):
        c = gst_fdpa(a[start:start + 64], b[start:start + 64], c,
                     alpha[start // block:(start + 64) // block],
                     beta[start // block:(start + 64) // block], nvfp4)
    return c
