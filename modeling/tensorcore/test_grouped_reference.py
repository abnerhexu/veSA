import unittest

from grouped_reference import fp4_half_units, gst_dot, gst_fdpa


class GroupedReferenceTests(unittest.TestCase):
    def test_all_fp4_encodings(self):
        expected = [0, 1, 2, 3, 4, 6, 8, 12]
        self.assertEqual([fp4_half_units(x) for x in range(8)], expected)
        self.assertEqual([fp4_half_units(x) for x in range(8, 16)], [-x for x in expected])

    def test_mxfp4_scale_block_is_not_compute_group(self):
        self.assertEqual(gst_fdpa([2]*64, [2]*64, 0, [127, 128], [127, 127]), 0x42c00000)

    def test_nvfp4_scale_blocks(self):
        self.assertEqual(gst_fdpa([2]*64, [2]*64, 0,
                                 [0x38, 0x40, 0x38, 0x40], [0x38]*4, True), 0x42c00000)

    def test_cancelling_group_retains_scale_exponent(self):
        a = [2]*8 + [10]*8 + [0]*48
        b = [2]*16 + [0]*48
        self.assertEqual(gst_fdpa(a, b, 0x3f800000, [200, 127], [200, 127]), 0)

    def test_nan_and_chained_blocks(self):
        self.assertEqual(gst_fdpa([0]*64, [0]*64, 0, [255, 127], [127, 127]), 0x7fffffff)
        self.assertEqual(gst_dot([2]*128, [2]*128, 0, [127]*4, [127]*4), 0x43000000)


if __name__ == "__main__":
    unittest.main()
