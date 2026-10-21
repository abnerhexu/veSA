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

The checked-in `results/` directory is the original synthetic arithmetic stress
corpus and remains useful for unit-level regression.  The primary experiment is
now `qwen_results/`, generated from the real BF16 checkpoint supplied outside
the repository.  Run it with:

```sh
../.venv/bin/python -m unittest -v \
  experiments/ws_sweeps/test_precision.py \
  experiments/ws_sweeps/test_qwen_corpus.py \
  experiments/ws_sweeps/test_qwen_outputs.py

../.venv/bin/python experiments/ws_sweeps/run_qwen_precision.py \
  --weights ../weight/model.safetensors-00001-of-00001.safetensors \
  --output experiments/ws_sweeps/qwen_results
```

The runner reads only four deterministic 16x256 slices with `numpy.memmap`; it
does not load the 1.75 GB checkpoint or instantiate the model.  The selected
language-model panels cover linear attention input projection, MLP down
projection, standard self-attention Q projection and MLP up projection.  For
each panel, seeded clipped-N(0,1) activations form A and the checkpoint slice,
transposed from `[out,in]`, forms B.  C is signed seeded normal data scaled to
the panel's dot-product magnitude.  Panel names, offsets, checkpoint SHA-256 and
all generation policies are fixed in `qwen_results/manifest.json`.

Both operands are quantized for every format.  FP8 uses nearest finite E4M3.
MXFP4 uses E2M1 values and per-row/per-column E8M0 scales over 32 K elements;
NVFP4 uses E2M1 and UE4M3 scales over 16.  A ceil scale avoids block clipping;
E2M1 and E4M3 conversion use a deterministic nearest-value rule.  This is a
documented experiment quantizer, not a claim about NVIDIA's software recipe.

Every structure runs on a logical 16x16 array at K=16, 32, 64, 128 and 256.
FP8 chains 32-element FDA blocks, with K=16 represented by a short block.  FP4
uses a 64-element Blackwell GDFS instruction: K=16/32 zero-pad unused lanes,
while K=128/256 chain rounded FP32 C between instructions.  Each CSV point
aggregates four matrices, or 1024 outputs.  Every candidate is compared with the
Blackwell configuration and an OS baseline that rounds every exact
low-precision product plus FP32 accumulator using IEEE FP32 RNE.

The CSVs report signed mean relative error (bias), mean/max absolute relative
error, mean absolute error, RMSE and bit-exact fraction.  Relative metrics omit
zero reference values and report their count explicitly, so positive and
negative errors cannot silently disappear behind an absolute-only metric.

Generated Qwen files (365 aggregate rows):

- `fp8_f_sweep.csv`: 25 rows, F=13/16/19/22/25, C-fused E4M3.
- `fp4_g_f_sweep.csv`: MXFP4/NVFP4, G=3..8 and F=26/29/32/35, GS=16.
- `fp4_gs_sweep.csv`: GS=12/14/16/18/20 at G=6, F=35.
- `c_coupling_sweep.csv`: C-fused and C-decoupled at F=6..10.
- `all_precision_sweeps.csv`: all 365 rows in one table.

The Qwen corpus shows FP8 error falling as F approaches Blackwell F=25.  In FP4,
changing F does not repair information discarded by a small G; G=6/7/8 is
bit-identical to Blackwell here, while G=3 has the largest error.  All swept GS
values happen to be bit-identical on these quantized panels at G=6/F=35, so this
corpus does not establish an accuracy distinction among GS choices.  The
synthetic stress corpus still exposes GS-dependent arithmetic differences.
C-decoupling materially reduces low-F error on the Qwen panels.

These are MMA output errors over real model weights, not an end-to-end Qwen task
accuracy evaluation, NVIDIA GPU differential, or reproduction of Kim et al.'s
synthesized area/energy.  Their G>=4 iso-accuracy statement is model-level;
nonzero pointwise MRE at G=4 is compatible with it.  Hardware cost for G/F/GS
and coupling is outside the A sweep unless separate parameterized RTL is added.
