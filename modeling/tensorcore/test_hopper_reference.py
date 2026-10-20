import unittest

from reference import fdpa, dot, pack


class HopperReferenceTests(unittest.TestCase):
    def test_c_only_subnormal_and_normal_e8m13(self):
        for c, expected in ((1, 0), (0x3ff, 0), (0x400, 0x400),
                            (0x80000001, 0), (0x3f8007ff, 0x3f800400),
                            (0xbf8007ff, 0xbf800400)):
            self.assertEqual(fdpa([0], [0], c, architecture="hopper"), expected)
        # A nonzero negative sum underflows to -0 only at the output converter;
        # a term already truncated to zero before summation produces +0.
        self.assertEqual(pack(-1, -149, output_fraction_bits=13), 0x80000000)

    def test_output_rounding_is_not_just_internal_f13(self):
        a, b = [0x38]*31 + [0x08], [0x38]*31 + [0x04]
        self.assertEqual(fdpa(a, b, 0, architecture="hopper"), 0x41f80000)
        self.assertEqual(fdpa(a, b, 0), 0x41f80040)
        self.assertEqual(pack(31*8192+1, -13), 0x41f80040)
        self.assertEqual(pack(31*8192+1, -13, output_fraction_bits=13), 0x41f80000)

    def test_block_end_rounding_before_next_c(self):
        a, b = [0x38]*31 + [0x08, 0xb8], [0x38]*31 + [0x04, 0x60]
        self.assertEqual(dot(a, b, 0, architecture="hopper"), 0xbf800000)
        self.assertNotEqual(dot(a, b, 0), 0xbf800000)

    def test_wgmma_half_zero_exponent(self):
        self.assertEqual(fdpa([0x0800, 1], [0x0c00, 1], 0, "fp16", True, "hopper"), 1)
        self.assertEqual(fdpa([0x0800, 1], [0x0c00, 1], 0, "fp16", True), 0)

    def test_canonical_nan_and_infinity(self):
        self.assertEqual(fdpa([0x7c], [0], 0, "e5m2", architecture="hopper"), 0x7fffffff)
        self.assertEqual(fdpa([0xfc], [0x3c], 0, "e5m2", architecture="hopper"), 0xff800000)


if __name__ == "__main__":
    unittest.main()
