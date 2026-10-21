import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "modeling/tensorcore"))

from qwen_corpus import (load_2d_slice, quantize_e4m3, quantize_fp4_rows,
                         read_safetensors_header)
from reference import FORMATS, decode
from grouped_reference import fp4_half_units, scale


class QwenCorpusTests(unittest.TestCase):
    def test_bf16_safetensors_slice_is_memory_addressable(self):
        values = np.arange(32, dtype=np.float32).reshape(4, 8) / 4
        raw = (values.view(np.uint32) >> 16).astype("<u2")
        header = {"weight": {"dtype": "BF16", "shape": [4, 8],
                             "data_offsets": [0, raw.nbytes]}}
        encoded = json.dumps(header, separators=(",", ":")).encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.safetensors"
            path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw.tobytes())
            data_start, tensors, _ = read_safetensors_header(path)
            got = load_2d_slice(path, data_start, tensors["weight"], 1, 2, 3, 4)
            np.testing.assert_array_equal(got, values[1:3, 3:7])

    def test_e4m3_quantizer_uses_finite_codes_and_both_signs(self):
        source = np.asarray([-500.0, -1.0, -0.1, 0.0, 0.1, 1.0, 500.0])
        codes = quantize_e4m3(source)
        decoded = [decode(int(code), FORMATS["e4m3"]) for code in codes]
        self.assertTrue(all(item[3] is None for item in decoded))
        values = [np.ldexp(item[0], item[1] - item[2]) for item in decoded]
        self.assertLess(values[0], 0)
        self.assertGreater(values[-1], 0)
        self.assertEqual(values[3], 0)

    def test_fp4_quantizer_respects_format_block_sizes(self):
        source = np.linspace(-7.0, 7.0, 64, dtype=np.float32).reshape(2, 32)
        for nvfp4, blocks in ((False, 1), (True, 2)):
            codes, scales = quantize_fp4_rows(source, nvfp4)
            self.assertEqual(codes.shape, source.shape)
            self.assertEqual(scales.shape, (2, blocks))
            self.assertTrue(all(-12 <= fp4_half_units(int(code)) <= 12
                                for code in codes.flat))
            self.assertTrue(all(scale(int(code), nvfp4)[3] is None
                                for code in scales.flat))


if __name__ == "__main__":
    unittest.main()
