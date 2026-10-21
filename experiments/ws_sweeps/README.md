# Weight-stationary width/K and precision sweeps

This experiment branch starts from Blackwell WS v2 commit `f197fdf`.  It keeps
the released `TensorMesh` unchanged and adds an isolated experiment target.
No FP32/TF32/FP64 A/B path is introduced.

## A. Physical width and K reuse

`WSReuseExperimentMesh` separates the 16-row physical K dimension from the
32-element E4M3 numerical fusion block.  For every FDA block the producer sends
the raw operands twice, following `io.expectedIndex`:

1. the exponent pass loads one 16-element physical chunk and propagates the
   narrow maximum lane;
2. after both physical chunks establish the final `e_max`, the significand pass
   reloads the same two chunks and propagates the fixed-point sum;
3. the block is normalized once, and the rounded FP32 result becomes C for the
   next block when K is 64, 128 or 256.

Only the current 16-element A chunk, two PE-local stationary B banks, per-output
`e_max`/sum/classification state and counters are retained.  The design contains
no product buffer and no full-K raw operand buffer.  Consequently, changing K
changes cycles and activity but not the synthesized circuit for a fixed width.

The handoff matrix contains five hardware points (`16x2`, `16x4`, `16x8`,
`16x16`, `16x32`) and four workloads per point (K=32/64/128/256).  The logical M
dimension is 16, FP32 C/D is retained, and useful work is `2*M*N*K` operations.

On the large-memory node, run each stage serially:

```sh
python3 experiments/ws_sweeps/prepare_area.py prepare --output out/ws-sweeps-area
python3 experiments/ws_sweeps/prepare_area.py rtl --output out/ws-sweeps-area
python3 experiments/ws_sweeps/prepare_area.py simulate --output out/ws-sweeps-area

python3 experiments/ws_sweeps/prepare_area.py synth \
  --output out/ws-sweeps-area \
  --yosys /path/to/oss-cad-suite/bin/yosys \
  --liberty /path/to/sky130_fd_sc_hd__tt_025C_1v80.lib

python3 experiments/ws_sweeps/prepare_area.py power \
  --output out/ws-sweeps-area \
  --openroad /path/to/oss-cad-suite/bin/openroad \
  --liberty /path/to/sky130_fd_sc_hd__tt_025C_1v80.lib \
  --tech-lef /path/to/sky130_fd_sc_hd__nom.tlef \
  --cell-lef /path/to/sky130_fd_sc_hd.lef

python3 experiments/ws_sweeps/prepare_area.py collect --output out/ws-sweeps-area
```

`prepare` writes source hashes, `hardware.csv`, `workloads.csv` and one checked
input/output vector per workload.  `simulate` verifies every output, records the
request-to-result cycle window, and writes both the raw 2 ps Chiseltest VCD and a
time-scaled 10 ns activity VCD.  `synth` maps each width once.  `power` evaluates
each K activity window on that width's mapped netlist.  `collect` produces
absolute area, power, energy, operations/cycle and operations/cycle/mm².

The power values are pre-layout OpenSTA estimates at an assumed 100 MHz with
0.1 ns input transition and 0.01 pF output load.  They are not Fmax, STA closure
or post-layout power results.  Inspect `annotation.txt` before accepting a point;
an unannotated or failed output check must not enter the final CSV.

## B. Precision sweeps

Run:

```sh
python3 -m unittest -v experiments/ws_sweeps/test_precision.py
python3 experiments/ws_sweeps/run_precision.py \
  --output experiments/ws_sweeps/results
```

The fixed corpus uses four signed 16x16 matrices per point.  FP8 uses K=256 and
chains 32-element FDA blocks; FP4 uses K=64.  Inputs cover both signs and a broad
finite exponent range, and C is signed and nonzero.  Every candidate is compared
against both the Blackwell configuration and an OS arithmetic baseline that
rounds each exact low-precision product plus accumulator with IEEE FP32 RNE.

The CSVs report signed mean relative error (bias), mean/max absolute relative
error, mean absolute error, RMSE and bit-exact fraction.  Relative metrics omit
zero reference values and report their count explicitly, so positive and
negative errors cannot silently disappear behind an absolute-only metric.

Generated files:

- `fp8_f_sweep.csv`: F=13/16/19/22/25, C-fused E4M3.
- `fp4_g_f_sweep.csv`: MXFP4/NVFP4, G=3..8 and F=26/29/32/35, GS=16.
- `fp4_gs_sweep.csv`: GS=12/14/16/18/20 at G=6, F=35.
- `c_coupling_sweep.csv`: C-fused and C-decoupled at F=6..10.
- `all_precision_sweeps.csv`: all 73 rows in one table.

The default corpus shows the intended qualitative effects.  FP8 error falls
monotonically as F approaches Blackwell F=25.  In FP4, changing F does not repair
the loss caused by small G; G=6/7/8 is bit-identical to Blackwell for this corpus,
while G=3 has large error.  GS=16 is the only swept group size that is bit-exact
to Blackwell because it is the architectural reference point.  C-decoupling
roughly halves the synthetic relative error over F=6..10 compared with C-fused.

These arithmetic-error results do not reproduce Kim et al.'s LLM task accuracy
or their synthesized area/energy numbers.  In particular, their statement that
G>=4 is iso-accurate is a model-level accuracy result; nonzero pointwise MRE at
G=4 is compatible with that conclusion.  Hardware cost for G/F/GS and coupling
is outside the A sweep unless separate parameterized RTL is added later.
