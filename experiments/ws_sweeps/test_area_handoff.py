import csv
import sys
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare_area import manifest, scale_vcd


class AreaHandoffTests(unittest.TestCase):
    def test_manifest_covers_five_widths_and_four_k_values(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            payload = manifest(output, 20261021)
            with (output / "hardware.csv").open() as stream:
                hardware = list(csv.DictReader(stream))
            with (output / "workloads.csv").open() as stream:
                workloads = list(csv.DictReader(stream))
            self.assertEqual(payload["hardware_points"], 5)
            self.assertEqual(payload["workload_points"], 20)
            self.assertEqual({int(row["columns"]) for row in hardware}, {2, 4, 8, 16, 32})
            self.assertEqual({int(row["k"]) for row in workloads}, {32, 64, 128, 256})
            self.assertTrue(all(Path(directory, row["vector"]).exists() for row in workloads))

    def test_vcd_scaling_converts_two_ps_cycle_to_ten_ns(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw.vcd"
            target = Path(directory) / "scaled.vcd"
            source.write_text("$timescale 1ps $end\n#0\n#1\n#2\n")
            scale_vcd(source, target)
            self.assertEqual(target.read_text(), "$timescale 1ps $end\n#0\n#5000\n#10000\n")


if __name__ == "__main__":
    unittest.main()
