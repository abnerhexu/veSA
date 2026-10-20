# First Q1 measurements

Measured on 2026-10-19 with the scripts in `../../`, on the experiment branch
based on Hopper commit `670b3d3c9d44eb2584e9b76de2f6ae1bd1eb3c59` plus the
uncommitted experiment sources identified by the numbered manifests.

Scope: M=N=2, E4M3 → FP32, core-only, SKY130 HD TT/25°C/1.8 V, no assumed
frequency. The IEEE baseline retains the released 32-term block scheduler and
uses a combinational FP32 RNE FMA in every PE. See `../../README.md` for methods.

| Design | Cell area (µm²) | Area / B0 | Cycles K=64 / 128 / 256 |
| --- | ---: | ---: | ---: |
| IEEE FMA | 74,198.66 | 1.000 | 72 / 144 / 288 |
| Hopper | 132,507.08 | 1.786 | 140 / 280 / 560 |
| Blackwell | 149,533.41 | 2.015 | 140 / 280 / 560 |

These are area/cycle tradeoffs, not clock-qualified wall-time comparisons.
Sequential cells account for 15.4%, 60.2%, and 55.5% of the respective areas;
this motivates examining the strict implementations' state/cache cost next.
It does not show that all sequential area is removable cache overhead.

The final suite checked 23 matrices per design, 69 matrices / 276 output words
in total, including a directed IEEE signed-zero case. Each strict expected
output was independently checked against MMA-Sim before RTL simulation.

`runs.csv` retains exploratory and final commands. Final simulation is run 10;
final synthesis runs are 11–13; final plotting is run 14. Initial runs 05–07
included synthesis assertions and are superseded by 11–13, which remove them
with `chformal -remove`. Run 09 added the signed-zero case. Numbered manifests
identify source and input hashes for each invocation; the RTL sources did not
change between the first and final measurements.

This directory preserves the small input/CSV/manifest evidence in the source
tree. Raw logs, generated RTL, mapped netlists, and PDF/SVG/PNG figures remain
in `../../../../out/dac-q1/initial/`. Reproduce with the command in
`../../README.md`. Power/energy rows explicitly remain unmeasured, and no
16 nm projection has been applied.
