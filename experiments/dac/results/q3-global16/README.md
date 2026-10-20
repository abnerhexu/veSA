# Q3: E4M3 OS scaling, fixed global M=N=16

IEEE FMA and Hopper/Blackwell slot reuse; array dimensions 2/4/8/16.
K=32/64/128/256/512 are performance points; K=1/31/33/65 and special values
provide additional correctness coverage. All 2566 complete tile matrices and
32080 output words passed raw-bit checks. Eight reset-aborted tasks are excluded.

The global-16 corpus was checked once against MMA-Sim 2.0.3 for both profiles
and libc fmaf for IEEE. Smaller arrays use slices of that checked corpus.
The reassembled A/B/C and all expected outputs match at every size, SHA256:
`2e28c45995818e3dbb0baebdbd1118640a7eeccca17463aede5abd3cce32d570`.

## Measured task cycles, K=256

| Array | IEEE FMA | Hopper reuse | Blackwell reuse |
| --- | ---: | ---: | ---: |
| 2×2 | 18558 | 20414 | 20414 |
| 4×4 | 5150 | 5614 | 5614 |
| 8×8 | 1542 | 1658 | 1658 |
| 16×16 | 512 | 541 | 541 |

The figure measures how the current L=32 OS implementations scale on one
global task. At K=256, 16×16 versus 2×2 gives 36.25× / 37.73× / 37.73×
cycle speedup, with area ratios 63.29 / 63.53 / 63.20. Throughput density
falls to 0.573 / 0.594 / 0.597 of each design's own 2×2 value.
Blackwell's effective MAC-slot occupancy falls from 80.3% to 47.3%.

Occupancy = global_M * global_N * K / (array_dim² * task_cycles), using one
useful accumulated product per PE/cycle as capacity. Collection/recomputed
products do not count twice. The heatmap uses Blackwell; Hopper task cycles
match at all sampled points. IEEE retains the released 32-term block scheduler,
so these data characterize the measured implementations, not an optimal IEEE core.

## Scope, provenance and reproduction

Task timing runs from first request acceptance to final result validity and
includes two handshake cycles between output tiles. Each tile retains full K.
No SRAM, DMA, host instruction or external bandwidth cost is included. Area is
native SKY130 HD TT/25°C/1.8 V synthesis area; 2×2/4×4 areas were reused with
source hashes, 8×8/16×16 areas were newly synthesized. Clock, power and 16 nm
projection are unmeasured. WS is not implemented.

All heavy tasks ran serially. JVM heap limit 1536 MiB; Mill --no-server -j 1.
8×8/16×16 simulations had already run/started before the user permitted native
make -j4, and retained -j1. New 2×2/4×4 simulations used -j4; the DAC test JVM
reports four processors to make Chiseltest pass that explicit job count.
The largest measured process RSS was 1651500 KiB (~1.58 GiB), not a process-tree sum.
Yosys/ABC retained a 4 GiB virtual-address limit.

From the submodule root, regenerate the CSV and PDF/SVG/PNG comparison:

```sh
MPLCONFIGDIR=out/dac-q3-global16/scaling/mpl-cache ../.venv/bin/python experiments/dac/scaling.py out/dac-q3-global16/scaling experiments/dac/results/q3-global16/dim2 experiments/dac/results/q3-global16/dim4 experiments/dac/results/q3-global16/dim8 experiments/dac/results/q3-global16/dim16
```

Per-stage commands and CPU retile method are in ../../README.md. Raw logs,
generated RTL and mapped netlists remain under out/dac-q3-global16/.
Numbered manifests retain the actual source hashes for each invocation.
