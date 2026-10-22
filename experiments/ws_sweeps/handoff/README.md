# Persisted WS sweep handoff artifacts

This directory preserves the useful outputs that previously existed only under
`/tmp`. It intentionally excludes Mill/Verilator object caches and other
regenerable build intermediates.

## `prepared/`

This is a complete seed-20261021 `prepare` output:

- `manifest.json` with hashes of the exact RTL, driver and arithmetic sources;
- `hardware.csv` with all five physical widths;
- `workloads.csv` with all 20 width/K points;
- one checked E4M3/FP32 `vector.tsv` for every point.

Copy its contents into a new output directory to begin at `rtl`. Every later
stage validates its source hashes before running.

## `local_smoke/`

`ws16x2-k64-runtime.csv` is the completed local activity smoke. It records 64
accepted input slices, 298 request-to-result cycles, 66 overlap cycles, 4096
operations and `result_check=passed`.

`rtl16x2/` preserves the generated 16x2/maxK=256 RTL smoke. The SystemVerilog
and FIRRTL are gzip-compressed; the annotation JSON is stored directly.

| File | Stored bytes | Stored SHA-256 |
| --- | ---: | --- |
| `WSReuseExperimentMesh.sv.gz` | 470248 | `b7305523bc2bbe300856188b769bf2a1a5529399031678d3501fa9b9428da5ff` |
| `WSReuseExperimentMesh.fir.gz` | 114737 | `3b8cbbcfbf33c30828f6272f6ce99d16bdd62e993cf0b11c9dd67f892cea988c` |
| `WSReuseExperimentMesh.anno.json` | 7606 | `8315de600968eea1260c295cd5f27474660cc1180c5d38369887eb67372f63e4` |
| `ws16x2-k64-runtime.csv` | 156 | `e7e1745208cffec22ad066f03a3cc53e8506a5faef1ff72b85f319f4043f754e` |

After decompression, the original RTL hashes are:

- SystemVerilog, 9,572,886 bytes:
  `66bd99c676409f481b4fb58bb65ce608ef9b3645d2814b98a4a877414d6b2dea`.
- FIRRTL, 1,737,810 bytes:
  `0bf59bc4cd1c2633306f3e37b644357b5ec6260d9d65e89ad45ad2b7d2038a70`.

This smoke export proves the generator path for one width. The large-node agent
must still generate, simulate and measure all five widths; it must not treat the
16x2 smoke as a complete A result.
