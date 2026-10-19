"""Integer-only paper776 T-FDPA oracle; FP32 is accumulator/output only.

No torch/numpy dependency. Exponents of subnormal inputs and zero terms follow
MMA-Sim 2.0.3 arithmetic/fdpa.py and nv_ptx/sim.py. Overflow follows that model
(infinity, including the RZ-FP32 conversion), not IEEE saturating round-to-zero.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Format:
    exponent: int
    fraction: int
    finite_only: bool = False

    @property
    def width(self):
        return 1 + self.exponent + self.fraction


FORMATS = {
    "e4m3": Format(4, 3, True), "e5m2": Format(5, 2),
    "fp16": Format(5, 10), "bf16": Format(8, 7),
}
FP32 = Format(8, 23)


def decode(bits, fmt):
    """Return (signed integer significand, exponent, fraction bits, special)."""
    assert 0 <= bits < 1 << fmt.width
    sign = -1 if bits >> (fmt.width - 1) else 1
    frac = bits & ((1 << fmt.fraction) - 1)
    exp = (bits >> fmt.fraction) & ((1 << fmt.exponent) - 1)
    if exp == (1 << fmt.exponent) - 1:
        if fmt.finite_only:
            if frac == (1 << fmt.fraction) - 1:
                return sign, 0, fmt.fraction, "nan"
        else:
            return sign, 0, fmt.fraction, "nan" if frac else "inf"
    bias = (1 << (fmt.exponent - 1)) - 1
    sig = frac if exp == 0 else (1 << fmt.fraction) | frac
    return sign * sig, max(exp, 1) - bias, fmt.fraction, None


def shift_rz(value, shift):
    magnitude = abs(value)
    magnitude = magnitude << shift if shift >= 0 else magnitude >> -shift
    return -magnitude if value < 0 else magnitude


def pack(value, power, half_output=False):
    """Pack exact value * 2**power: RZ-FP32 or RNE-FP16."""
    fmt = FORMATS["fp16"] if half_output else FP32
    sign = (1 << (fmt.width - 1)) if value < 0 else 0
    value = abs(value)
    if not value:
        return 0
    bias = (1 << (fmt.exponent - 1)) - 1
    leading_exp = value.bit_length() - 1 + power
    unit_exp = max(leading_exp, 1 - bias) - fmt.fraction
    shift = unit_exp - power
    if shift > 0:
        sig, remainder = divmod(value, 1 << shift)
        if half_output:
            halfway = 1 << (shift - 1)
            sig += remainder > halfway or (remainder == halfway and bool(sig & 1))
    else:
        sig = value << -shift
    if sig >= 1 << (fmt.fraction + 1):
        sig >>= 1
        unit_exp += 1
    exponent = unit_exp + fmt.fraction
    if exponent > bias:
        return sign | (((1 << fmt.exponent) - 1) << fmt.fraction)
    if sig < 1 << fmt.fraction:
        return sign | sig
    return sign | ((exponent + bias) << fmt.fraction) | (sig & ((1 << fmt.fraction) - 1))


def fdpa(a, b, c, format_name="e4m3", half_output=False):
    """A single Blackwell fused block, including C and final conversion."""
    fmt = FORMATS[format_name]
    assert not (format_name == "bf16" and half_output), "No native BF16 -> FP16 MMA"
    limit = 32 if fmt.width == 8 else 16
    assert 1 <= len(a) == len(b) <= limit
    zero_exp = -22 if half_output else -133
    terms = []
    infinity_signs = set()
    nan = False
    for ai, bi in zip(a, b):
        sa, ea, fa, special_a = decode(ai, fmt)
        sb, eb, fb, special_b = decode(bi, fmt)
        if "nan" in (special_a, special_b):
            nan = True
        elif "inf" in (special_a, special_b):
            if sa == 0 or sb == 0:
                nan = True
            else:
                infinity_signs.add((sa < 0) ^ (sb < 0))
        else:
            terms.append((sa * sb, ea + eb if sa * sb else zero_exp, fa + fb))
    sc, ec, fc, special_c = decode(c, FORMATS["fp16"] if half_output else FP32)
    if special_c == "nan":
        nan = True
    elif special_c == "inf":
        infinity_signs.add(sc < 0)
    else:
        terms.append((sc, ec if sc else zero_exp, fc))
    width = 16 if half_output else 32
    if nan or len(infinity_signs) > 1:
        return (1 << (width - 1)) - 1
    if infinity_signs:
        return (int(next(iter(infinity_signs))) << (width - 1)) | (0x7c00 if half_output else 0x7f800000)
    maximum = max(e for _, e, _ in terms)
    total = sum(shift_rz(s, 25 - f + e - maximum) for s, e, f in terms)
    return pack(total, maximum - 25, half_output)


def dot(a, b, c, format_name="e4m3", half_output=False):
    """Sequential fused blocks; only block outputs become the next C."""
    assert len(a) == len(b) and len(a) > 0
    length = 32 if FORMATS[format_name].width == 8 else 16
    for start in range(0, len(a), length):
        c = fdpa(a[start:start + length], b[start:start + length], c, format_name, half_output)
    return c
