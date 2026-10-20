# B2: collect the next block during replay

Measured on 2026-10-19, on `experiments/dac-q1`, based on the same Hopper release
commit as `../q1-initial/`. Source and input hashes are in the numbered manifests.

Each PE has two banks of raw E4M3 A/B. One bank collects the next block while
the other replays exact products. Per-bank product maxima and special flags are
collected independently of C. Replay starts only after the preceding output has
been converted and its C has been included in the common exponent.

The mesh forwards continuously across fusion boundaries and freezes forwarding
when a PE's banks fill. This B2 changes both cache representation and scheduling;
it is not an isolated single-variable scheduling ablation.

| Design | SKY130 area (µm²) | K=64 cycles | K=128 cycles | K=256 cycles |
| --- | ---: | ---: | ---: | ---: |
| IEEE B0 (reused) | 74,198.66 | 72 | 144 | 288 |
| Hopper B1 (reused) | 132,507.08 | 140 | 280 | 560 |
| Blackwell B1 (reused) | 149,533.41 | 140 | 280 | 560 |
| Hopper overlap B2 | 220,896.86 | 105 | 175 | 315 |
| Blackwell overlap B2 | 233,433.88 | 105 | 175 | 315 |

At K=256, B2 reduces cycles by 43.75%, increasing area by 66.71% (Hopper) and
56.11% (Blackwell). Operations/cycle/area improves by 6.64% and 13.88% relative
to each B1 under this area/cycle scope. This is not a clock-qualified speedup or
an energy result.

For continuous inputs, the first input of the second block is accepted at cycle
33. Both variants pass all 23 unchanged first-batch matrices plus a bubbled /
backpressured K=65 matrix and a K=65 restart after reset during overlap. The
suite therefore checks 50 complete matrices / 200 output words, plus two aborted
tasks. Historical B0/B1 designs were not rerun; `baseline-reuse.json` records
their source directory and raw statistics/runtime hashes.

Raw artifacts and PDF/SVG/PNG plots are in
`../../../../out/dac-q1/overlap/`; reproduction is documented in `../../README.md`.
No SRAM/RoCC, other arithmetic profiles, power, or node scaling are claimed by
this measurement. The production release mesh is unchanged; this is the B2
experimental core.
