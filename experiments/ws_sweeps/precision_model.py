"""Configurable integer models for the WS precision sweeps.

The models operate on raw low-precision encodings.  Every truncation is an
integer magnitude shift followed by sign restoration, matching the RZ rule in
the Tensor reference model.  FP32 is an accumulator/output interface only.
"""
from dataclasses import dataclass
from pathlib import Path
import math
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "modeling/tensorcore"))
from reference import FP32, FORMATS, decode, pack, shift_rz  # noqa: E402
from grouped_reference import scale  # noqa: E402


@dataclass(frozen=True)
class Term:
    significand: int
    exponent: int
    fraction: int


def fp32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def fp32_value(bits):
    return struct.unpack("<f", struct.pack("<I", bits & 0xffffffff))[0]


def finite_term(bits, fmt):
    significand, exponent, fraction, special = decode(bits, fmt)
    if special is not None:
        raise ValueError("precision sweeps use finite values only")
    return Term(significand, exponent, fraction)


def product_terms(a, b, format_name="e4m3", zero_exponent=-133):
    fmt = FORMATS[format_name]
    terms = []
    for left_bits, right_bits in zip(a, b):
        left = finite_term(left_bits, fmt)
        right = finite_term(right_bits, fmt)
        product = left.significand * right.significand
        terms.append(Term(product,
                          left.exponent + right.exponent if product else zero_exponent,
                          left.fraction + right.fraction))
    return terms


def aligned_sum(terms, fractional_bits):
    """Return (integer sum, binary power, maximum exponent)."""
    if not terms:
        return 0, -fractional_bits, -133
    maximum = max(term.exponent for term in terms)
    total = sum(shift_rz(term.significand,
                         fractional_bits - term.fraction + term.exponent - maximum)
                for term in terms)
    return total, maximum - fractional_bits, maximum


def exact_add(left_value, left_power, right_value, right_power):
    power = min(left_power, right_power)
    return ((left_value << (left_power - power)) +
            (right_value << (right_power - power))), power


def pack_rne_fp32(value, power):
    """Pack exact value*2**power as IEEE FP32 round-to-nearest-even."""
    sign = 0x80000000 if value < 0 else 0
    value = abs(value)
    if value == 0:
        return sign
    fraction = 23
    bias = 127
    leading_exp = value.bit_length() - 1 + power
    unit_exp = max(leading_exp, 1 - bias) - fraction
    shift = unit_exp - power
    if shift > 0:
        significand, remainder = divmod(value, 1 << shift)
        halfway = 1 << (shift - 1)
        significand += remainder > halfway or (remainder == halfway and bool(significand & 1))
    else:
        significand = value << -shift
    if significand >= 1 << (fraction + 1):
        significand >>= 1
        unit_exp += 1
    exponent = unit_exp + fraction
    if exponent > bias:
        return sign | 0x7f800000
    if significand < 1 << fraction:
        return sign | significand
    return sign | ((exponent + bias) << fraction) | (significand & ((1 << fraction) - 1))


def configurable_fdpa(a, b, c, fractional_bits, c_fused=True,
                      format_name="e4m3"):
    products = product_terms(a, b, format_name)
    c_term = finite_term(c, FP32)
    if c_fused:
        terms = products + [Term(c_term.significand,
                                 c_term.exponent if c_term.significand else -133,
                                 c_term.fraction)]
        total, power, _ = aligned_sum(terms, fractional_bits)
        return pack(total, power)

    dot, dot_power, _ = aligned_sum(products, fractional_bits)
    combined, power = exact_add(dot, dot_power, c_term.significand,
                                c_term.exponent - c_term.fraction)
    return pack(combined, power)


def configurable_dot(a, b, c, fractional_bits, c_fused=True,
                     format_name="e4m3", fusion_length=32):
    assert len(a) == len(b) and len(a) > 0
    for start in range(0, len(a), fusion_length):
        c = configurable_fdpa(a[start:start + fusion_length],
                              b[start:start + fusion_length], c,
                              fractional_bits, c_fused, format_name)
    return c


def os_dot(a, b, c, format_name="e4m3"):
    """Sequential low-precision-product/FP32-RNE FMA baseline."""
    for product in product_terms(a, b, format_name):
        accumulator = finite_term(c, FP32)
        combined, power = exact_add(product.significand,
                                    product.exponent - product.fraction,
                                    accumulator.significand,
                                    accumulator.exponent - accumulator.fraction)
        c = pack_rne_fp32(combined, power)
    return c


def fp4_decode(bits):
    magnitude = bits & 7
    if magnitude == 0:
        return 0, 0, 1
    if magnitude < 4:
        significand, exponent = magnitude, 0
    else:
        significand, exponent = (2 if magnitude % 2 == 0 else 3), magnitude // 2 - 1
    return (-significand if bits & 8 else significand), exponent, 1


def fp4_unscaled_terms(a, b):
    terms = []
    for index, (left, right) in enumerate(zip(a, b)):
        left_sig, left_exp, left_frac = fp4_decode(left)
        right_sig, right_exp, right_frac = fp4_decode(right)
        significand = left_sig * right_sig
        terms.append(Term(significand,
                          left_exp + right_exp if significand else -139,
                          left_frac + right_frac))
    return terms


def fp4_product_terms(a, b, alpha, beta, nvfp4=False):
    block = 16 if nvfp4 else 32
    assert len(a) == len(b)
    assert len(alpha) == len(beta) == math.ceil(len(a) / block)
    terms = []
    for index, raw in enumerate(fp4_unscaled_terms(a, b)):
        sa, ea, fa, special_a = scale(alpha[index // block], nvfp4)
        sb, eb, fb, special_b = scale(beta[index // block], nvfp4)
        if special_a is not None or special_b is not None:
            raise ValueError("precision sweeps use finite scales only")
        significand = raw.significand * sa * sb
        terms.append(Term(significand,
                          raw.exponent + ea + eb if significand else -139,
                          raw.fraction + fa + fb))
    return terms


def configurable_gdfs(a, b, c, alpha, beta, group_bits=6,
                      fractional_bits=35, group_size=16, nvfp4=False,
                      c_fused=True):
    """GDFS with independent intra-group G, inter-group F and group size."""
    assert group_bits >= 0 and fractional_bits >= 0 and group_size > 0
    block = 16 if nvfp4 else 32
    products = fp4_unscaled_terms(a, b)
    groups = []
    for start in range(0, len(products), group_size):
        end = min(start + group_size, len(products))
        segment = start
        scaled_segments = []
        while segment < end:
            scale_index = segment // block
            segment_end = min(end, (scale_index + 1) * block)
            total, power, maximum = aligned_sum(products[segment:segment_end], group_bits)
            sa, ea, fa, special_a = scale(alpha[scale_index], nvfp4)
            sb, eb, fb, special_b = scale(beta[scale_index], nvfp4)
            if special_a is not None or special_b is not None:
                raise ValueError("precision sweeps use finite scales only")
            scaled = total * sa * sb
            scaled_segments.append(Term(
                scaled, maximum + ea + eb if scaled else -139,
                group_bits + fa + fb))
            assert power == maximum - group_bits
            segment = segment_end
        if len(scaled_segments) == 1:
            groups.append(scaled_segments[0])
        else:
            # GS and the storage scale block are independent. If a group spans
            # a scale boundary, combine its scaled sub-dots at G plus the scale
            # significand precision before emitting exactly one inter-group term.
            scale_fraction = max(term.fraction - group_bits
                                 for term in scaled_segments)
            total, _, maximum = aligned_sum(
                scaled_segments, group_bits + scale_fraction)
            groups.append(Term(total, maximum if total else -139,
                               group_bits + scale_fraction))
    accumulator = finite_term(c, FP32)
    if c_fused:
        groups.append(Term(accumulator.significand,
                           accumulator.exponent if accumulator.significand else -139,
                           accumulator.fraction))
        total, power, _ = aligned_sum(groups, fractional_bits)
        return pack(total, power)
    dot, dot_power, _ = aligned_sum(groups, fractional_bits)
    combined, power = exact_add(dot, dot_power, accumulator.significand,
                                accumulator.exponent - accumulator.fraction)
    return pack(combined, power)


def configurable_gdfs_dot(a, b, c, alpha, beta, group_bits=6,
                          fractional_bits=35, group_size=16, nvfp4=False,
                          fusion_length=64, c_fused=True):
    """Chain 64-way GDFS blocks, zero-padding only the final short block.

    Blackwell exposes a 64-element FP4 instruction.  K=16 and K=32 experiments
    therefore model unused lanes as exact zero, while K>64 passes the rounded
    FP32 result of one instruction as C to the next instruction.
    """
    assert len(a) == len(b) and len(a) > 0
    assert fusion_length == 64
    block = 16 if nvfp4 else 32
    expected_scales = math.ceil(len(a) / block)
    assert len(alpha) == len(beta) == expected_scales
    neutral = 0x38 if nvfp4 else 127
    for start in range(0, len(a), fusion_length):
        end = min(start + fusion_length, len(a))
        chunk_a = list(a[start:end])
        chunk_b = list(b[start:end])
        chunk_a.extend([0] * (fusion_length - len(chunk_a)))
        chunk_b.extend([0] * (fusion_length - len(chunk_b)))
        scale_start = start // block
        scale_end = math.ceil(end / block)
        chunk_alpha = list(alpha[scale_start:scale_end])
        chunk_beta = list(beta[scale_start:scale_end])
        scales_per_instruction = fusion_length // block
        chunk_alpha.extend([neutral] * (scales_per_instruction - len(chunk_alpha)))
        chunk_beta.extend([neutral] * (scales_per_instruction - len(chunk_beta)))
        c = configurable_gdfs(chunk_a, chunk_b, c, chunk_alpha, chunk_beta,
                              group_bits, fractional_bits, group_size, nvfp4,
                              c_fused)
    return c


def os_fp4_dot(a, b, c, alpha, beta, nvfp4=False):
    for product in fp4_product_terms(a, b, alpha, beta, nvfp4):
        accumulator = finite_term(c, FP32)
        combined, power = exact_add(product.significand,
                                    product.exponent - product.fraction,
                                    accumulator.significand,
                                    accumulator.exponent - accumulator.fraction)
        c = pack_rne_fp32(combined, power)
    return c


def error_metrics(candidate, reference, prefix):
    assert len(candidate) == len(reference)
    signed_relative = []
    signed_error_over_magnitude = []
    absolute_relative = []
    squared = []
    absolute = []
    exact = 0
    finite = 0
    zero_reference = 0
    for got_bits, ref_bits in zip(candidate, reference):
        got = fp32_value(got_bits)
        ref = fp32_value(ref_bits)
        if got_bits == ref_bits:
            exact += 1
        if not math.isfinite(got) or not math.isfinite(ref):
            continue
        finite += 1
        delta = got - ref
        absolute.append(abs(delta))
        squared.append(delta * delta)
        if ref == 0.0:
            zero_reference += 1
        else:
            signed_relative.append(delta / ref)
            signed_error_over_magnitude.append(delta / abs(ref))
            absolute_relative.append(abs(delta) / abs(ref))
    return {
        f"{prefix}_finite_samples": finite,
        f"{prefix}_zero_reference_samples": zero_reference,
        f"{prefix}_signed_mean_relative_error": sum(signed_relative) / len(signed_relative) if signed_relative else 0.0,
        f"{prefix}_mean_signed_error_over_abs_reference": sum(signed_error_over_magnitude) / len(signed_error_over_magnitude) if signed_error_over_magnitude else 0.0,
        f"{prefix}_mean_absolute_relative_error": sum(absolute_relative) / len(absolute_relative) if absolute_relative else 0.0,
        f"{prefix}_max_absolute_relative_error": max(absolute_relative, default=0.0),
        f"{prefix}_mean_absolute_error": sum(absolute) / len(absolute) if absolute else 0.0,
        f"{prefix}_rmse": math.sqrt(sum(squared) / len(squared)) if squared else 0.0,
        f"{prefix}_bit_exact_fraction": exact / len(candidate),
    }
