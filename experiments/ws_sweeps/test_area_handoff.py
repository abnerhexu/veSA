import csv
import json
import sys
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare_area import collect, manifest, scale_vcd, validate_manifest
from ws_breakdown import (aggregate_area, aggregate_power, classify_operators,
                          rules, write_table)


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
            self.assertEqual(payload["breakdown"]["categories"], rules()["categories"])
            self.assertFalse(payload["breakdown"]["proportional_allocation"])
            self.assertEqual(validate_manifest(output), payload)

            stored = json.loads((output / "manifest.json").read_text())
            first = next(iter(stored["source_sha256"]))
            stored["source_sha256"][first] = "0" * 64
            (output / "manifest.json").write_text(json.dumps(stored))
            with self.assertRaisesRegex(RuntimeError, "do not match this checkout"):
                validate_manifest(output)

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
            write_table(root / "area-breakdown.csv", [
                {"category": category, "cells": 1 if index == 0 else 0,
                 "area_um2": 100000.0 if index == 0 else 0.0,
                 "area_percent": 100.0 if index == 0 else 0.0}
                for index, category in enumerate(rules()["categories"])])
            write_table(root / "storage-area-breakdown.csv", [
                {"storage_detail": detail, "cells": 0, "area_um2": 0.0,
                 "storage_percent": 0.0, "design_percent": 0.0}
                for detail in rules()["storage_details"]])
            (point / "runtime.csv").write_text(
                "cycles,operations,overlap_cycles,result_check\n"
                "200,2048,0,passed\n")
            (point / "power.json").write_text(json.dumps(
                {"Total": {"internal": 0.2, "switching": 0.2,
                           "leakage": 0.1, "total": 0.5}}))
            (point / "annotation.txt").write_text("vcd 100\nunannotated 0\n")
            write_table(point / "power-breakdown.csv", [
                {"category": category, "cells": 1 if index == 0 else 0,
                 "area_um2": 100000.0 if index == 0 else 0.0,
                 "internal_w": 0.2 if index == 0 else 0.0,
                 "switching_w": 0.2 if index == 0 else 0.0,
                 "leakage_w": 0.1 if index == 0 else 0.0,
                 "total_w": 0.5 if index == 0 else 0.0,
                 "power_percent": 100.0 if index == 0 else 0.0}
                for index, category in enumerate(rules()["categories"])])
            write_table(point / "storage-power-breakdown.csv", [
                {"storage_detail": detail, "cells": 0, "area_um2": 0.0,
                 "internal_w": 0.0, "switching_w": 0.0,
                 "leakage_w": 0.0, "total_w": 0.0,
                 "power_percent": 0.0, "design_power_percent": 0.0}
                for detail in rules()["storage_details"]])
            collect(output, (2,), (32,))
            with (output / "summary.csv").open() as stream:
                row = next(csv.DictReader(stream))
            self.assertAlmostEqual(float(row["throughput_tops"]), 0.001024)
            self.assertAlmostEqual(float(row["throughput_tops_per_mm2"]), 0.01024)
            self.assertAlmostEqual(float(row["power_density_w_per_mm2"]), 5.0)
            with (output / "power-breakdown.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 7)

    def test_breakdown_area_and_power_close_over_disjoint_cells(self):
        configuration = rules()
        cells = [
            {"instance": "u_mul", "category": "multiply_decode", "storage_detail": "",
             "area_um2": 2.0},
            {"instance": "u_reg", "category": "storage_interconnect",
             "storage_detail": "stationary_weight_storage", "area_um2": 3.0},
        ]
        area, storage = aggregate_area(cells, configuration)
        self.assertAlmostEqual(sum(float(row["area_um2"]) for row in area), 5.0)
        self.assertEqual(next(row for row in storage
                              if row["storage_detail"] == "stationary_weight_storage")["area_um2"], 3.0)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            write_table(directory / "cell-power.tsv", [
                {"instance": "u_mul", "internal_w": 0.10, "switching_w": 0.20,
                 "leakage_w": 0.01, "total_w": 0.31},
                {"instance": "u_reg", "internal_w": 0.20, "switching_w": 0.30,
                 "leakage_w": 0.02, "total_w": 0.52},
            ], delimiter="\t")
            (directory / "power.json").write_text(json.dumps({"Total": {
                "internal": 0.30, "switching": 0.50, "leakage": 0.03, "total": 0.83}}))
            power, storage_power, combined = aggregate_power(
                cells, directory / "cell-power.tsv", directory / "power.json", configuration)
            self.assertEqual(len(power), 7)
            self.assertEqual(len(storage_power), len(configuration["storage_details"]))
            self.assertEqual(len(combined), 2)
            self.assertAlmostEqual(sum(row["total_w"] for row in power), 0.83)

    def test_pre_map_ownership_separates_arithmetic_state_and_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            rtl = Path(directory) / "tiny.sv"
            rtl.write_text(
                "wire product; // @[src/TensorArithmetic.scala 40:29]\n"
                "wire mystery;\n")
            span_product = f"{rtl}:1.1-1.60"
            span_unknown = f"{rtl}:2.1-2.20"
            design = {"modules": {"WSReuseExperimentPE": {
                "attributes": {},
                "netnames": {
                    "product": {"bits": [10], "attributes": {"src": span_product}},
                    "weights_0": {"bits": [20], "attributes": {}}},
                "cells": {
                    "mul": {"type": "$mul", "attributes": {"src": span_product},
                            "connections": {"Y": [10]}, "port_directions": {"Y": "output"}},
                    "weight_ff": {"type": "$dff", "attributes": {"src": span_unknown},
                                  "connections": {"Q": [20]}, "port_directions": {"Q": "output"}},
                    "unknown": {"type": "$add", "attributes": {"src": span_unknown},
                                "connections": {"Y": [30]}, "port_directions": {"Y": "output"}}
                }}}}
            ledger = {row["operator"]: row for row in classify_operators(design, rtl)}
            self.assertEqual(ledger["mul"]["category"], "multiply_decode")
            self.assertEqual(ledger["weight_ff"]["category"], "storage_interconnect")
            self.assertEqual(ledger["weight_ff"]["storage_detail"],
                             "stationary_weight_storage")
            self.assertEqual(ledger["unknown"]["category"], "shared_other")


if __name__ == "__main__":
    unittest.main()
