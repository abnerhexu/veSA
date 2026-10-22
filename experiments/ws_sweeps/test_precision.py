import sys
from pathlib import Path
import random
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "modeling/tensorcore"))

from precision_model import (configurable_dot, configurable_fdpa,
                             configurable_gdfs, configurable_gdfs_dot,
                             error_metrics, fp32_bits, os_dot)
from reference import dot, fdpa
from grouped_reference import gst_fdpa


class PrecisionSweepModelTests(unittest.TestCase):
    def test_blackwell_fp8_configuration_matches_release_oracle(self):
        a = [0x38, 0xb8, 0x41, 0xc2] * 8
        b = [0x39, 0x3a, 0xba, 0x42] * 8
        c = fp32_bits(-0.75)
        self.assertEqual(configurable_fdpa(a, b, c, 25, True), fdpa(a, b, c))
        self.assertEqual(configurable_dot(a[:16], b[:16], c, 25, True),
                         dot(a[:16], b[:16], c))
        self.assertEqual(configurable_dot(a * 2, b * 2, c, 25, True),
                         dot(a * 2, b * 2, c))

    def test_blackwell_fp4_configuration_matches_release_oracle(self):
        for nvfp4, neutral in ((False, 127), (True, 0x38)):
            a = [(index * 5) & 15 for index in range(64)]
            b = [(index * 7 + 3) & 15 for index in range(64)]
            scales = 4 if nvfp4 else 2
            alpha = [neutral] * scales
            beta = [neutral] * scales
            c = fp32_bits(0.375)
            self.assertEqual(configurable_gdfs(a, b, c, alpha, beta,
                                                6, 35, 16, nvfp4),
                             gst_fdpa(a, b, c, alpha, beta, nvfp4))
        a = [(index * 3 + 1) & 15 for index in range(64)]
        b = [(index * 11 + 2) & 15 for index in range(64)]
        for nvfp4, alpha, beta in (
                (False, [123, 132], [130, 121]),
                (True, [0x30, 0x41, 0x28, 0x4a], [0x3c, 0x25, 0x48, 0x32])):
            c = fp32_bits(-0.625)
            self.assertEqual(configurable_gdfs(a, b, c, alpha, beta,
                                                6, 35, 16, nvfp4),
                             gst_fdpa(a, b, c, alpha, beta, nvfp4))

    def test_short_and_chained_fp4_dot(self):
        a = [(index * 5) & 15 for index in range(128)]
        b = [(index * 7 + 3) & 15 for index in range(128)]
        for nvfp4, neutral, block in ((False, 127, 32), (True, 0x38, 16)):
            c = fp32_bits(-0.25)
            short = configurable_gdfs_dot(
                a[:16], b[:16], c, [neutral], [neutral], 6, 35, 16, nvfp4)
            padded = configurable_gdfs(
                a[:16] + [0] * 48, b[:16] + [0] * 48, c,
                [neutral] * (64 // block), [neutral] * (64 // block),
                6, 35, 16, nvfp4)
            self.assertEqual(short, padded)
            chained = configurable_gdfs_dot(
                a, b, c, [neutral] * (128 // block),
                [neutral] * (128 // block), 6, 35, 16, nvfp4)
            expected = gst_fdpa(a[64:], b[64:],
                                gst_fdpa(a[:64], b[:64], c,
                                         [neutral] * (64 // block),
                                         [neutral] * (64 // block), nvfp4),
                                [neutral] * (64 // block),
                                [neutral] * (64 // block), nvfp4)
            self.assertEqual(chained, expected)

    def test_non_power_of_two_groups_cross_scale_blocks(self):
        a = [7 if index % 3 else 15 for index in range(64)]
        b = [6 if index % 5 else 14 for index in range(64)]
        alpha = [124, 130]
        beta = [129, 122]
        outputs = [configurable_gdfs(a, b, 0, alpha, beta, 6, 35, gs, False)
                   for gs in (12, 14, 16, 18, 20)]
        self.assertEqual(len(outputs), 5)
        self.assertTrue(all(0 <= value < 1 << 32 for value in outputs))

    def test_cross_scale_group_is_not_split_into_inter_group_terms(self):
        rng = random.Random(1)
        different = False
        for _ in range(16):
            a = [rng.randrange(16) for _ in range(64)]
            b = [rng.randrange(16) for _ in range(64)]
            alpha = [rng.randrange(120, 135) for _ in range(2)]
            beta = [rng.randrange(120, 135) for _ in range(2)]
            combined = configurable_gdfs(a, b, 0, alpha, beta, 3, 8, 20, False)
            split = configurable_gdfs(a, b, 0, alpha, beta, 3, 8, 16, False)
            different |= combined != split
        self.assertTrue(different)

    def test_gdfs_c_decoupling_preserves_c_at_second_alignment(self):
        rng = random.Random(2)
        different = False
        for trial in range(16):
            a = [rng.randrange(16) for _ in range(64)]
            b = [rng.randrange(16) for _ in range(64)]
            c = fp32_bits((-.5 if trial % 2 else .5) * 2 ** rng.randrange(-8, 9))
            alpha = [rng.randrange(120, 135) for _ in range(2)]
            beta = [rng.randrange(120, 135) for _ in range(2)]
            fused = configurable_gdfs(a, b, c, alpha, beta, 6, 6, 16, False, True)
            decoupled = configurable_gdfs(a, b, c, alpha, beta, 6, 6, 16, False, False)
            different |= fused != decoupled
        self.assertTrue(different)

    def test_c_decoupled_preserves_small_dot_added_to_large_c(self):
        a = [0x08] * 32
        b = [0x08] * 32
        c = fp32_bits(1.0)
        fused = configurable_fdpa(a, b, c, 6, True)
        decoupled = configurable_fdpa(a, b, c, 6, False)
        self.assertNotEqual(fused, decoupled)
        self.assertEqual(decoupled, os_dot(a, b, c))

    def test_os_fma_rounding_and_signs(self):
        a = [0x38, 0xb8]
        b = [0x38, 0x38]
        self.assertEqual(os_dot(a, b, fp32_bits(0.5)), fp32_bits(0.5))

    def test_signed_relative_error_uses_signed_reference(self):
        metrics = error_metrics(
            [fp32_bits(0.5), fp32_bits(-0.5)],
            [fp32_bits(1.0), fp32_bits(-1.0)], "x")
        self.assertEqual(metrics["x_signed_mean_relative_error"], -0.5)
        self.assertEqual(metrics["x_mean_signed_error_over_abs_reference"], 0.0)


if __name__ == "__main__":
    unittest.main()
