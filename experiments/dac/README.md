# Q1: E4M3 core area and cycles

Run from the Gemmini submodule, using the parent Python environment:

```sh
../.venv/bin/python experiments/dac/run.py all --output out/dac-q1/initial
```

The runner executes vector generation, seven RTL elaborations, one serial
simulation suite, seven synthesis tasks, and plotting in that order. Individual
stages (`vectors`, `rtl`, `simulate`, `synth`, `plot`) can reuse the same output
directory; `--design ieee|hopper|blackwell|hopper-overlap|blackwell-overlap`
selects one point. `--design overlap` selects the two B2 variants together.
`--design reuse` selects the two slot-reuse variants. The selected IDs are
forwarded through `Q1_DESIGNS` so simulation runs exactly the requested pair.
Mill workers require local sockets, even with `--no-server`.

The experiments use a 2×2 output-stationary array, E4M3 raw A/B, FP32 C/D, one
task in flight, continuously available inputs and an immediately available
output consumer. This is **core-only**: the interface payload includes C/D but
there is no SRAM, DMA, host tiler, or memory bandwidth model in this experiment.
No production arithmetic or release test source is changed.

## Implementations and numerical checks

- B0 (`ieee`): exact E4M3-to-FP32 conversion followed by HardFloat FP32 RNE FMA
  on every product. FP32 is an internal accumulator format; the external A/B
  ports are still 8 bits. NaN outputs use `0x7fc00000`.
- B1 (`hopper`, `blackwell`): the actual released `TensorMesh` / `TensorPE`
  implementation, configured for each architecture's E4M3 → FP32 profile.
- B2 (`hopper-overlap`, `blackwell-overlap`): two raw E4M3 A/B banks per PE,
  with independent collection and replay. Each bank records its product maximum
  and special-value flags. Replay recomputes exact products from the raw bank,
  incorporates the preceding block's rounded C into the final common exponent,
  and performs the same truncation, sum, and output conversion as B1. The mesh
  forwards a continuous wavefront across block boundaries, stalling globally
  only when a PE's operand-bank capacity is exhausted.
- B2 slot reuse (`hopper-reuse-overlap`, `blackwell-reuse-overlap`): retain two
  sets of logical block metadata but store operands in one physical bank.
  While replay reads slot i, the next block may overwrite slots up to i, using
  the old register value on a same-edge read/write. The writer stalls before
  any unread slot. Waiting for a partly collected block permits normal writes;
  waiting for a full block starts replay before permitting overwrite. C and
  the replay arithmetic are shared with the original dual-bank B2.
- The IEEE mesh copies the released 32-term block/wavefront scheduler, replacing
  only the PE. Thus B0 retains the same inter-block control overhead, although
  IEEE accumulation does not numerically require 32-term boundaries. It has one
  combinational FMA per PE and no clock-feasibility claim.

`vectors.py` combines the fixed release E4M3 matrices, small encoding cases, and
fixed-seed M=N=2, K=64/128/256 tasks. It checks each B1 expected output against
MMA-Sim 2.0.3 before simulation. The independent B0 oracle calls libc `fmaf`;
ordinary separate Python multiplication/addition is not used as an IEEE FMA
oracle. All three RTL designs check every output bit and the returned tag.
Historical release cases keep their bubble/backpressure coverage in the release
tests; this measurement suite feeds them without artificial stalls.
For B2 the suite also tests input bubbles, output backpressure, and reset while
the second block is partially collected during first-block replay. The extra
cases are separately named in the CSV and excluded from throughput plots.

## Measurement scope

Cycles start immediately after the matrix request acceptance edge and end at
the first boundary where the result is valid. Output-consumption and reset
cycles are excluded. `input_cycles` is the number of accepted K slices;
`non_input_cycles` is the remaining request-to-result time, not a detailed
replay/control attribution. Effective payload bytes are `4*K + 32` (four FP8
values per K slice and four FP32 C plus four FP32 D values).

Synthesis uses the fixed Volare SKY130 revision
`c6d73a35f524070e85faff4a6a9eef49553ebc2b`, HD TT/25°C/1.8 V Liberty and the
local OSS CAD Suite Yosys/ABC. It flattens each PE for constant propagation
across arithmetic submodules, retaining the four-PE hierarchy. Low-power
isolation cells (`sky130_fd_sc_hd__lpflow_*`) are excluded from mapping.
Simulation assertions are removed from the synthesis input with `chformal
-remove`; assertions remain enabled in the RTL simulation.
The two area categories are deliberately coarse: PE arithmetic/local state and
mesh control/forwarding. They do not claim a fine-grained arithmetic breakdown.

Results are post-synthesis standard-cell areas in µm² and RTL cycles. No
frequency is assumed, and there is no slack/Fmax or layout experiment. The
plots report operations/cycle and operations/cycle/mm². Power/energy and 16 nm
projections remain unmeasured until their estimation methods are fixed.

## Outputs and resource use

The output directory contains `manifest.json`, `runs.csv`, author-checked input
vectors, generated RTL, raw synthesis/test logs, tool resource reports, mapped
netlists, `ppa.csv`, `runtime.csv`, `breakdown.csv`, `summary.csv` and PDF/SVG/
300 dpi PNG plots. `energy.csv` explicitly marks the first batch unmeasured.
Each invocation keeps its own numbered manifest. Failed commands remain in `runs.csv`; rerunning a stage
adds a new log instead of deleting the earlier failure.

All heavy tasks run sequentially. Test JVMs are limited to 1536 MiB and one
active processor; native builds use `-j1`. Yosys and its ABC child each inherit
a 4 GiB virtual-address-space limit. `/usr/bin/time -v` reports maximum process
RSS per command; that metric is not the sum of all concurrently resident worker
processes. No VCD is emitted in the first area/cycle batch.

## Reusing the first batch for the overlap comparison

`results/q1-initial/` preserves the original B0/B1 evidence. The measured B2
comparison is in `out/dac-q1/overlap/`, with a small permanent snapshot under
`results/q1-overlap/`. Its `baseline-reuse.json` identifies the original three
measurements; those unchanged designs were not rerun. The input vectors are
identical to the final first-batch vectors.

After the first-batch vectors, three `*-runtime.csv` and `*-stat.json` files
are copied into a new output directory, the new points can be measured with:

```sh
../.venv/bin/python experiments/dac/run.py rtl --design overlap --output out/dac-q1/overlap
../.venv/bin/python experiments/dac/run.py simulate --design overlap --output out/dac-q1/overlap
../.venv/bin/python experiments/dac/run.py synth --design overlap --output out/dac-q1/overlap
../.venv/bin/python experiments/dac/run.py plot --output out/dac-q1/overlap
```

All measurements here validate only the E4M3/FP32 experimental core. B2 is not
connected to Gemmini's production SRAM/RoCC path, and its parameterized class
does not imply that other input/output formats have been validated.

## Slot reuse comparison

`results/q1-reuse/` and `out/dac-q1/reuse/` preserve the next Q2 measurement.
Copy the same vectors and five prior `*-runtime.csv` / `*-stat.json` files from
the overlap batch into the new directory, then run the four stages above with
`--design reuse` for RTL/simulation/synthesis and the new output path for all
stages. `baseline-reuse.json` identifies the reused data and hashes.

The comparison figure shows B0, dual-bank B2, and slot-reuse B2. `summary.csv`
retains the original serial B1 points as well. This experiment uses registers
with old-value read/write semantics; it does not establish an SRAM macro port
mapping or read-during-write behavior.

## Q3: fixed global task across array sizes

`--dim` parameterizes the experimental wrapper and IEEE mesh; production
arithmetic remains unchanged. `--design q3` selects IEEE FMA and both slot-reuse
profiles. Use `--problem-dim 4` for identical 4×K by K×4 global inputs at each
array size. The 2×2 array runs four output tiles serially; the 4×4 array runs
one. K=1/31/32/33/65 and special values supplement the K=64/128/256 performance
tasks. Each tile retains the full K dimension and its FP32 C initial value.

For example, run these stages serially, changing `--dim` and output directory
for the other size:

```sh
../.venv/bin/python experiments/dac/run.py vectors --design q3 --dim 4 --problem-dim 4 --output out/dac-q3/dim4
../.venv/bin/python experiments/dac/run.py rtl --design q3 --dim 4 --problem-dim 4 --output out/dac-q3/dim4
../.venv/bin/python experiments/dac/run.py simulate --design q3 --dim 4 --problem-dim 4 --output out/dac-q3/dim4
../.venv/bin/python experiments/dac/run.py synth --design q3 --dim 4 --problem-dim 4 --output out/dac-q3/dim4
../.venv/bin/python experiments/dac/run.py plot --design q3 --dim 4 --problem-dim 4 --output out/dac-q3/dim4
../.venv/bin/python experiments/dac/scaling.py out/dac-q3/scaling out/dac-q3/dim2 out/dac-q3/dim4
```

`runtime.csv` retains individual tile windows. `task-runtime.csv` directly
measures first request acceptance through final result validity, including
the result-consumption and next-request edges between tiles. With this driver
that equals the sum of tile windows plus two cycles per tile boundary.
It excludes host instruction overhead and any SRAM/DMA or external bandwidth
cost. Inputs are continuously available. The scaling script reassembles A/B/C
and all expected output matrices and requires an identical global-task hash
across sizes; it writes `scaling.csv`, `sources.json`, and PDF/SVG/PNG figures.

The unchanged 2×2 areas can be reused from Q1 with their source paths/hashes
recorded in `baseline-reuse.json`; the new fixed-global-task vectors and RTL
cycles must still be measured at both sizes. Parameter choices 8/16 enable
future experiments; they are not a claim of completed validation. All reported
areas remain native SKY130 synthesis results, without timing or node scaling.
