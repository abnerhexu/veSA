"""Memory-bounded Qwen weight sampling and low-precision quantization."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np

from precision_model import fp32_bits
from reference import FORMATS, decode


DEFAULT_TENSORS = (
    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
    "model.language_model.layers.0.mlp.down_proj.weight",
    "model.language_model.layers.11.self_attn.q_proj.weight",
    "model.language_model.layers.23.mlp.up_proj.weight",
)


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple
    offsets: tuple


def read_safetensors_header(path):
    """Read metadata only; tensor payloads remain on disk."""
    path = Path(path)
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"not a safetensors file: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        header = json.loads(stream.read(header_length))
    tensors = {
        name: TensorInfo(name, item["dtype"], tuple(item["shape"]),
                         tuple(item["data_offsets"]))
        for name, item in header.items() if name != "__metadata__"
    }
    return 8 + header_length, tensors, header.get("__metadata__", {})


def _bf16_to_float32(raw):
    words = np.asarray(raw, dtype=np.uint16).astype(np.uint32) << 16
    return words.view(np.float32)


def load_2d_slice(path, data_start, info, row_start, rows, column_start, columns):
    if len(info.shape) != 2:
        raise ValueError(f"{info.name} is not a matrix: {info.shape}")
    if row_start < 0 or column_start < 0 or row_start + rows > info.shape[0] \
            or column_start + columns > info.shape[1]:
        raise ValueError(f"slice exceeds {info.name} shape {info.shape}")
    dtype_map = {"BF16": "<u2", "F16": "<f2", "F32": "<f4"}
    if info.dtype not in dtype_map:
        raise ValueError(f"unsupported weight dtype {info.dtype} for {info.name}")
    matrix = np.memmap(path, mode="r", dtype=dtype_map[info.dtype],
                       offset=data_start + info.offsets[0], shape=info.shape)
    raw = np.asarray(matrix[row_start:row_start + rows,
                            column_start:column_start + columns])
    values = _bf16_to_float32(raw) if info.dtype == "BF16" else raw.astype(np.float32)
    return np.array(values, dtype=np.float32, copy=True)


def _format_value(bits, format_name):
    significand, exponent, fraction, special = decode(bits, FORMATS[format_name])
    if special is not None:
        return None
    return math.ldexp(significand, exponent - fraction)


def _encoding_table(entries):
    # Prefer positive zero when two encodings represent the same value.
    unique = {}
    for value, code in entries:
        unique.setdefault(float(value), int(code))
    values = np.asarray(sorted(unique), dtype=np.float32)
    codes = np.asarray([unique[float(value)] for value in values], dtype=np.uint8)
    return values, codes


E4M3_VALUES, E4M3_CODES = _encoding_table(
    (_format_value(code, "e4m3"), code) for code in range(256)
    if _format_value(code, "e4m3") is not None
)
FP4_HALF_UNITS = (0, 1, 2, 3, 4, 6, 8, 12)
FP4_VALUES, FP4_CODES = _encoding_table(
    ((-1 if code & 8 else 1) * FP4_HALF_UNITS[code & 7] / 2.0, code)
    for code in range(16)
)
POSITIVE_E4M3 = np.asarray(sorted(value for value in E4M3_VALUES if value > 0),
                           dtype=np.float32)
POSITIVE_E4M3_CODES = np.asarray(
    [E4M3_CODES[np.searchsorted(E4M3_VALUES, value)] for value in POSITIVE_E4M3],
    dtype=np.uint8)


def _nearest_codes(values, levels, codes):
    values = np.asarray(values, dtype=np.float32)
    right = np.searchsorted(levels, values, side="left")
    right = np.clip(right, 0, len(levels) - 1)
    left = np.clip(right - 1, 0, len(levels) - 1)
    # Ties select the lower numeric value, a deterministic policy recorded in
    # the manifest. Exact Blackwell conversion tie behavior is not claimed.
    choose_right = np.abs(values - levels[right]) < np.abs(values - levels[left])
    indices = np.where(choose_right, right, left)
    return codes[indices]


def quantize_e4m3(values):
    """Quantize finite float values to nearest finite E4M3 encoding."""
    if not np.all(np.isfinite(values)):
        raise ValueError("E4M3 experiment inputs must be finite")
    return _nearest_codes(values, E4M3_VALUES, E4M3_CODES).astype(np.uint8)


def _e8m0_scale(maximum):
    if maximum == 0.0:
        return 127, 1.0
    exponent = math.ceil(math.log2(maximum / 6.0))
    exponent = min(127, max(-127, exponent))
    return exponent + 127, math.ldexp(1.0, exponent)


def _ue4m3_scale(maximum):
    if maximum == 0.0:
        return 0x38, 1.0
    required = maximum / 6.0
    index = int(np.searchsorted(POSITIVE_E4M3, required, side="left"))
    index = min(index, len(POSITIVE_E4M3) - 1)
    return int(POSITIVE_E4M3_CODES[index]), float(POSITIVE_E4M3[index])


def quantize_fp4_rows(values, nvfp4=False):
    """Block-quantize the last dimension; return E2M1 codes and scale codes."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("FP4 quantizer expects a finite rank-2 matrix")
    block = 16 if nvfp4 else 32
    codes = np.empty(values.shape, dtype=np.uint8)
    scale_codes = np.empty((values.shape[0], math.ceil(values.shape[1] / block)),
                           dtype=np.uint8)
    for row in range(values.shape[0]):
        for block_index, start in enumerate(range(0, values.shape[1], block)):
            segment = values[row, start:start + block]
            maximum = float(np.max(np.abs(segment), initial=0.0))
            scale_code, scale_value = (_ue4m3_scale(maximum) if nvfp4
                                       else _e8m0_scale(maximum))
            scale_codes[row, block_index] = scale_code
            codes[row, start:start + len(segment)] = _nearest_codes(
                segment / scale_value, FP4_VALUES, FP4_CODES)
    return codes, scale_codes


def deterministic_panel_offsets(name, shape, seed, max_k=256, n=16):
    if shape[0] < n or shape[1] < max_k:
        raise ValueError(f"tensor {name} shape {shape} cannot provide {n}x{max_k}")
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    row_blocks = (shape[0] - n) // n + 1
    column_blocks = (shape[1] - max_k) // max_k + 1
    row_start = (int.from_bytes(digest[:8], "little") % row_blocks) * n
    column_start = (int.from_bytes(digest[8:16], "little") % column_blocks) * max_k
    return row_start, column_start


def make_panel(path, data_start, info, seed, max_k=256, m=16, n=16):
    row_start, column_start = deterministic_panel_offsets(
        info.name, info.shape, seed, max_k, n)
    # Safetensors stores W[out,in]; X @ W.T uses B[k,n].
    weight = load_2d_slice(path, data_start, info, row_start, n,
                           column_start, max_k).T
    panel_seed = int.from_bytes(hashlib.sha256(
        f"activation:{seed}:{info.name}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(panel_seed)
    activation = np.clip(rng.standard_normal((m, max_k)), -6.0, 6.0).astype(np.float32)
    c_unit = rng.standard_normal((m, n)).astype(np.float32)
    return {
        "name": info.name, "dtype": info.dtype, "shape": info.shape,
        "row_start": row_start, "column_start": column_start,
        "a": activation, "b": weight, "c_unit": c_unit,
    }


def fp32_c_matrix(panel, k):
    scale = math.sqrt(k) * float(np.std(panel["b"][:k], dtype=np.float64))
    values = np.asarray(panel["c_unit"] * scale, dtype=np.float32)
    return [[fp32_bits(float(value)) for value in row] for row in values]
