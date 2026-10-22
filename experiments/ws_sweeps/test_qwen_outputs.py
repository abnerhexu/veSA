import csv
import hashlib
import json
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "qwen_results"


class QwenOutputTests(unittest.TestCase):
    def test_committed_result_matrix_and_hashes(self):
        manifest = json.loads((RESULTS / "manifest.json").read_text())
        self.assertEqual(manifest["systolic_array"], "16x16")
        self.assertEqual(manifest["K"], [16, 32, 64, 128, 256])
        self.assertEqual(len(manifest["panels"]), 4)
        expected_rows = {
            "fp8_f_sweep.csv": 25,
            "fp4_g_f_sweep.csv": 240,
            "fp4_gs_sweep.csv": 50,
            "c_coupling_sweep.csv": 150,
            "all_precision_sweeps.csv": 465,
        }
        for name, count in expected_rows.items():
            path = RESULTS / name
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), count, name)
            self.assertEqual({int(row["K"]) for row in rows},
                             {16, 32, 64, 128, 256})
            self.assertTrue(all(int(row["M"]) == int(row["N"]) == 16
                                for row in rows))
            self.assertTrue(all("vs_blackwell_mean_signed_error_over_abs_reference" in row
                                for row in rows))
            if name == "c_coupling_sweep.csv":
                self.assertEqual({row["format"] for row in rows},
                                 {"e4m3", "mxfp4", "nvfp4"})
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, manifest["files"][name])


if __name__ == "__main__":
    unittest.main()
