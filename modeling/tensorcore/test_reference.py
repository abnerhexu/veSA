import unittest

from reference import decode, dot, fdpa, FORMATS, pack, shift_rz


class ReferenceTests(unittest.TestCase):
    def test_paper_discrepancy(self):
        a = [0xf000, 0xb800, 0xb400, 0xb000] + [0] * 12
        b = [0x6400, 0x3c00, 0x3c00, 0x3c00] + [0] * 12
        self.assertEqual(fdpa(a, b, 0x4b000000, "fp16"), 0xbf400000)

    def test_signed_truncation(self):
        self.assertEqual(shift_rz(-7, -2), -1)

    def test_half_ties_and_subnormal(self):
        self.assertEqual(pack(2049, -11, True), 0x3c00)
        self.assertEqual(pack(2051, -11, True), 0x3c02)
        self.assertEqual(pack(1, -24, True), 1)
        self.assertEqual(pack(-1, -25, True), 0x8000)

    def test_e4m3_extended_finite(self):
        self.assertEqual(decode(0x7e, FORMATS["e4m3"]), (14, 8, 3, None))
        self.assertEqual(fdpa([0x7e], [0x38], 0), 0x43e00000)
        self.assertEqual(fdpa([0xff], [0x38], 0), 0x7fffffff)

    def test_special_values(self):
        self.assertEqual(fdpa([0x7c], [0], 0, "e5m2"), 0x7fffffff)
        self.assertEqual(fdpa([0xfc], [0x3c], 0, "e5m2"), 0xff800000)
        self.assertEqual(fdpa([0x7c], [0x3c], 0xff800000, "e5m2"), 0x7fffffff)

    def test_chained_blocks(self):
        self.assertEqual(dot([0x38]*65, [0x38]*65, 0), 0x42820000)


if __name__ == "__main__":
    unittest.main()
