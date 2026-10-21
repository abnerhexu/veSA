import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class PrecisionSweepOutputTests(unittest.TestCase):
    def test_cli_emits_complete_sweeps_and_reference_points(self):
        script = Path(__file__).with_name("run_precision.py")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            subprocess.run((sys.executable, script, "--output", output,
                            "--trials", "1", "--m", "4", "--n", "4"),
                           check=True, stdout=subprocess.DEVNULL)
            with (output / "all_precision_sweeps.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 73)
            fp8 = next(row for row in rows if row["experiment"] == "fp8_fractional_bits"
                       and row["F"] == "25")
            self.assertEqual(float(fp8["vs_blackwell_bit_exact_fraction"]), 1.0)
            for fmt in ("mxfp4", "nvfp4"):
                blackwell = next(row for row in rows if row["experiment"] == "fp4_g_f"
                                 and row["format"] == fmt and row["G"] == "6"
                                 and row["F"] == "35")
                self.assertEqual(float(blackwell["vs_blackwell_bit_exact_fraction"]), 1.0)
            for fractional_bits in map(str, range(6, 11)):
                fused = next(row for row in rows if row["experiment"] == "c_coupling"
                             and row["coupling"] == "c_fused" and row["F"] == fractional_bits)
                decoupled = next(row for row in rows if row["experiment"] == "c_coupling"
                                 and row["coupling"] == "c_decoupled" and row["F"] == fractional_bits)
                self.assertLess(float(decoupled["vs_blackwell_mean_absolute_relative_error"]),
                                float(fused["vs_blackwell_mean_absolute_relative_error"]))


if __name__ == "__main__":
    unittest.main()
