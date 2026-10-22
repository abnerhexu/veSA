import csv
import json
import sys
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare_area import collect, manifest, scale_vcd


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
            self.assertTrue(all(row["operand_passes"] == "1" for row in hardware))
            self.assertTrue(all(row["segment_buffers"] == "2" for row in hardware))
            self.assertTrue(all(row["weights_per_pe"] == "4" for row in hardware))
            self.assertTrue(all(Path(directory, row["vector"]).exists() for row in workloads))

    def test_vcd_scaling_converts_two_ps_cycle_to_ten_ns(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw.vcd"
            target = Path(directory) / "scaled.vcd"
            source.write_text("$timescale 1ps $end\n#0\n#1\n#2\n")
            scale_vcd(source, target)
            self.assertEqual(target.read_text(), "$timescale 1ps $end\n#0\n#5000\n#10000\n")

    def test_collect_rejects_incomplete_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest(output, 20261021)
            with self.assertRaisesRegex(FileNotFoundError, "missing synthesized area"):
                collect(output, (2,), (32,))

    def test_collect_emits_absolute_and_per_area_throughput(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            root = output / "ws16x2"
            point = root / "k32"
            point.mkdir(parents=True)
            (root / "stat.json").write_text(json.dumps(
                {"design": {"area": 100000.0}}))
            (point / "runtime.csv").write_text(
                "cycles,operations,overlap_cycles,result_check\n"
                "200,2048,0,passed\n")
            (point / "power.json").write_text(json.dumps(
                {"Total": {"total": 0.5}}))
            (point / "annotation.txt").write_text("sequential coverage 100%\n")
            collect(output, (2,), (32,))
            with (output / "summary.csv").open() as stream:
                row = next(csv.DictReader(stream))
            self.assertAlmostEqual(float(row["throughput_tops"]), 0.001024)
            self.assertAlmostEqual(float(row["throughput_tops_per_mm2"]), 0.01024)
            self.assertAlmostEqual(float(row["power_density_w_per_mm2"]), 5.0)


if __name__ == "__main__":
    unittest.main()
