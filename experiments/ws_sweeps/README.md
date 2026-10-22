# Weight-stationary width/K and precision sweeps

This experiment branch starts from Blackwell WS v2 commit `f197fdf`.  It keeps
the released `TensorMesh` unchanged and adds an isolated experiment target.
No FP32/TF32/FP64 A/B path is introduced.

## Handoff contract

Use the `experiments/ws-sweeps` branch, tagged `ws-sweeps-handoff-v2`, and run
commands from the veSA submodule root.  Do not run this flow from another branch:
the prepared-input manifest hashes `build.mill`, the experiment RTL/test/driver
and `TensorArithmetic.scala`, and every stage after `prepare` rejects a source
mismatch.

The repository already contains the completed B results and local A validation
artifacts.  A large node only needs to run the five-width RTL/activity/PPA flow;
it must not rerun B unless the weight file, quantizer or arithmetic model changes.

Required tools and resource policy:

- Java 17 and Mill 1.3.0 (pinned by `build.mill`).
- Python 3 with NumPy for B verification/regeneration; A's handoff driver uses
  only the standard library and the checked-in integer reference model.
- Verilator; local validation used 5.052.  The wrapper under
  `scripts/tensorcore-verilator` handles the Chiseltest compatibility define.
- Yosys with ABC, an OpenROAD build providing OpenSTA/VCD power commands, and a
  mutually consistent Liberty, technology LEF and standard-cell LEF.  The flow
  was written for `sky130_fd_sc_hd` TT/1.8 V/25 C.
- Run one experiment process at a time.  Mill always uses `--no-server -j 1`,
  test JVM heap is limited to 1536 MiB, and native Verilator make uses `-j 4`.
- Reserve enough disk for 20 raw and time-scaled VCDs plus five mapped netlists.

Before starting, verify the checkout and tools:

```sh
git status --short
git describe --tags --exact-match
java -version
mill --version
verilator --version

mill --no-server -j 1 wsExperiments.compile
python3 -m unittest discover -v -s experiments/ws_sweeps -p 'test_*.py'
```

The first command must show no experiment-branch modifications, and the tag
must be `ws-sweeps-handoff-v2`.  Preserve any unrelated work by using a separate
Git worktree rather than switching a dirty checkout.

## A. Physical width and K reuse

`WSReuseExperimentMesh` separates the 16-row physical K dimension from the
32-element E4M3 numerical fusion block.  Two segment buffers each hold one
32-term FDA as two physical chunks.  Every PE therefore has one multiplier lane
and four stationary weight bytes: two chunks in each of two segment banks.  A
is retained only in two 32-term left-edge buffers; products and complete-K
vectors are never retained.

The producer sends each raw K slice once.  The narrow lane computes a
product-only maximum for the next segment while the multiplier/fixed-sum lane
replays the current segment.  When replay begins, the boundary logic combines
the precomputed product maximum with the newly normalized prior C exponent to
obtain the true `e_max`.  The two physical chunks share that maximum and the
segment is normalized exactly once.  For K=64/128/256, the rounded FP32 result
becomes C for the next 32-term segment.  The runtime CSV records cycles where
both lanes are active and tests require nonzero overlap for K>32.

The handoff matrix contains five hardware points (`16x2`, `16x4`, `16x8`,
`16x16`, `16x32`) and four workloads per point (K=32/64/128/256).  The logical M
dimension is 16, FP32 C/D is retained, and useful work is `2*M*N*K` operations.

On the large-memory node, define absolute tool/PDK paths and run each stage
serially.  `WS_OUT` may be on a large scratch filesystem:

```sh
export WS_OUT="$PWD/out/ws-sweeps-area"
export YOSYS=/path/to/oss-cad-suite/bin/yosys
export OPENROAD=/path/to/OpenROAD/bin/openroad
export LIBERTY=/path/to/sky130_fd_sc_hd__tt_025C_1v80.lib
export TECH_LEF=/path/to/sky130_fd_sc_hd__nom.tlef
export CELL_LEF=/path/to/sky130_fd_sc_hd.lef

test -x "$YOSYS" && test -x "$OPENROAD"
test -f "$LIBERTY" && test -f "$TECH_LEF" && test -f "$CELL_LEF"

python3 experiments/ws_sweeps/prepare_area.py prepare --output "$WS_OUT"
python3 experiments/ws_sweeps/prepare_area.py rtl --output "$WS_OUT"
python3 experiments/ws_sweeps/prepare_area.py simulate --output "$WS_OUT"

python3 experiments/ws_sweeps/prepare_area.py synth \
  --output "$WS_OUT" --yosys "$YOSYS" --liberty "$LIBERTY"

python3 experiments/ws_sweeps/prepare_area.py power \
  --output "$WS_OUT" --openroad "$OPENROAD" \
  --liberty "$LIBERTY" --tech-lef "$TECH_LEF" --cell-lef "$CELL_LEF"

python3 experiments/ws_sweeps/prepare_area.py collect --output "$WS_OUT"
```

`handoff/prepared/` contains the already generated seed-20261021 manifest and
all 20 checked vectors.  To resume without regenerating them, copy that directory
to an empty output directory instead of running `prepare`:

```sh
mkdir -p "$WS_OUT"
cp -a experiments/ws_sweeps/handoff/prepared/. "$WS_OUT"/
python3 experiments/ws_sweeps/prepare_area.py rtl --output "$WS_OUT"
```

All stages accept `--columns 2|4|8|16|32`; `simulate` and `power` also accept
`--k 32|64|128|256`.  These selectors support recovery of one failed point.
Rerunning a selected stage overwrites that point's generated output.  Run the
final `collect` without selectors so it requires all five area results and all
20 checked power points.

`prepare` writes source hashes, `hardware.csv`, `workloads.csv` and one checked
input/output vector per workload.  `simulate` verifies every output, records the
request-to-result cycle window, and writes both the raw 2 ps Chiseltest VCD and a
time-scaled 10 ns activity VCD.  `synth` assigns every generic RTL operator to a
disjoint stage before lowering and ABC, preserves those stage modules, maps each
width once, and checks the sum of physical leaf-cell areas against Yosys `stat`.
`power` evaluates each K activity window on the same mapped netlist, records
every physical leaf-cell's OpenSTA power, and checks the leaf sums against
OpenSTA `Total`.  `collect` produces
absolute area, power, energy, operations/cycle, TOPS and both normalized values
per mm².  Collection fails instead of silently omitting a point if any selected
area, checked runtime, power result, breakdown or activity annotation is missing.

### Area and power breakdown

The primary breakdown uses seven mutually exclusive categories.  The same
physical leaf instance belongs to the same category in area and all four K
power runs for a width:

- `multiply_decode`: E4M3 decode, significand multiplication, product sign,
  classification and product exponent generation.
- `exponent_align`: product/C maximum selection, exponent difference, variable
  shift and magnitude-first RZ truncation.
- `fixed_sum`: the fixed-point partial-sum adder logic.
- `normalize_convert`: leading-zero/priority logic, normalization, FP32 packing
  and NaN/Inf result selection.
- `storage_interconnect`: every sequential cell plus register-data selection and
  array/feedback routing.
- `control`: FSM, counters, valid/ready, bank/chunk selection and tags.
- `shared_other`: combinational gates with ambiguous or unresolved ownership.

Sequential cells are always charged to `storage_interconnect`, even when they
pipeline an arithmetic stage.  Shared logic is reported explicitly and is never
distributed in proportion to another category.  Because the stage hierarchy is
established before technology mapping, the reported total is the constrained,
category-preserving implementation itself.  Use this total for all width/K
comparisons from this handoff; do not combine it with a separately flattened
synthesis result.

`storage_interconnect` also has an auxiliary, disjoint sub-breakdown:

- `stationary_weight_storage`: four stationary B bytes per PE.
- `activation_buffers`: the two 32-term left-edge A buffers.
- `accumulator_state`: C/D, product maxima, replay maxima, fixed sums and
  NaN/Inf state.
- `feedback_pipeline`: the exponent return path from column bottom to top.
- `pipeline_interconnect`: PE exponent/sum wave pipeline registers.
- `control_state`: registered FSM, sequence, cycle and result state.
- `data_routing`: combinational register reads, muxes and cross-stage routing.
- `unclassified_storage`: physical storage that cannot be assigned to one of
  the named storage roles; it must remain visible rather than being reallocated.

The final root files are:

- `area-breakdown.csv`: 35 rows, seven categories for each of five widths.
- `storage-area-breakdown.csv`: 40 rows, eight storage details per width.
- `power-breakdown.csv`: 140 rows, seven categories for every width/K point.
- `storage-power-breakdown.csv`: 160 rows, eight storage details per width/K.
- `summary.csv`: 20 total PPA/throughput rows.

Area rows contain physical cell count, µm² and percentage.  Power rows contain
internal, switching, leakage and total watts plus percentage.  Per-width
`cells.csv` is the area ownership ledger; each workload retains
`cell-power.tsv`, `cells-power.csv`, both point breakdowns and total
`power.json`.  `breakdown-rules.json`, `operator-ownership.csv` and
`stage-ownership.json` preserve the classification policy and audit trail.
The synthesis stage also retains `raw-mapped.json`, repaired `mapped.json`,
`pin-repair.json` and one `*-abc-reference.json` per mapped stage.  These files
make the installed Yosys/ABC Liberty-port repair auditable; the repair matches
exact ABC cell ordinals and net correspondences and never guesses pin order.

`synth` records the Yosys executable/version/hash and Liberty hash in
`environment.json`; `power` adds the OpenROAD executable/version/hash and all
three PDK-file hashes.  Keep this file with `summary.csv`.  The expected final
inventory is five `stat.json`/`mapped.json`/`mapped.v`/`cells.csv` sets, 20
`runtime.csv` files, 20 scaled activity VCDs, 20 nonempty
`annotation.txt`/`power.json`/`cell-power.tsv` sets, the four aggregate breakdown
CSVs above, and a 20-row `summary.csv`.  Every runtime row must say
`result_check=passed`; K>32 rows must have `overlap_cycles > 0`.

The power values are pre-layout OpenSTA estimates at an assumed 100 MHz with
0.1 ns input transition and 0.01 pF output load.  They are not Fmax, STA closure
or post-layout power results.  Inspect `annotation.txt` before accepting a point;
an unannotated or failed output check must not enter the final CSV.
The driver enforces the explicit `unannotated 0` result before accepting power.

The checked-in `handoff/local_smoke/` directory preserves the completed
16x2/K=64 runtime and compressed 16x2 RTL export.  It is validation evidence,
not a substitute for generating and measuring all five widths on the large
node.  Its README records original and compressed hashes.

## B. Precision sweeps

The checked-in `results/` directory is the original synthetic arithmetic stress
corpus and remains useful for unit-level regression.  The primary experiment is
now `qwen_results/`, generated from the real BF16 checkpoint supplied outside
the repository.  The checked-in CSV and manifest are the deliverable.  To verify
them without rerunning the 465-point sweep:

```sh
python3 -m unittest -v experiments/ws_sweeps/test_qwen_outputs.py
```

Only regenerate B when its inputs or implementation change.  The full
regeneration command is:

```sh
python3 -m unittest -v \
  experiments/ws_sweeps/test_precision.py \
  experiments/ws_sweeps/test_qwen_corpus.py \
  experiments/ws_sweeps/test_qwen_outputs.py

python3 experiments/ws_sweeps/run_qwen_precision.py \
  --weights /absolute/path/to/model.safetensors-00001-of-00001.safetensors \
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

Generated Qwen files (465 aggregate rows):

- `fp8_f_sweep.csv`: 25 rows, F=13/16/19/22/25, C-fused E4M3.
- `fp4_g_f_sweep.csv`: MXFP4/NVFP4, G=3..8 and F=26/29/32/35, GS=16.
- `fp4_gs_sweep.csv`: GS=12/14/16/18/20 at G=6, F=35.
- `c_coupling_sweep.csv`: FP8/MXFP4/NVFP4 C-fused and C-decoupled at F=6..10.
- `all_precision_sweeps.csv`: all 465 rows in one table.

The Qwen corpus shows FP8 error falling as F approaches Blackwell F=25.  In FP4,
changing F does not repair information discarded by a small G; G=6/7/8 is
bit-identical to Blackwell here, while G=3 has the largest error.  GS is kept
independent of the MXFP4/NVFP4 storage scale block: a group crossing a format
block first forms scaled sub-dots, then emits one G-precision group term.  This
exposes small but nonzero GS-dependent errors on the Qwen panels.
C-decoupling materially reduces low-F error in all three formats.

`signed_mean_relative_error` uses `(candidate-reference)/reference`.
`mean_signed_error_over_abs_reference` separately reports the direction of the
numeric delta normalized by reference magnitude.  MARE continues to use the
absolute delta and absolute reference.

These are MMA output errors over real model weights, not an end-to-end Qwen task
accuracy evaluation, NVIDIA GPU differential, or reproduction of Kim et al.'s
synthesized area/energy.  Their G>=4 iso-accuracy statement is model-level;
nonzero pointwise MRE at G=4 is compatible with it.  Hardware cost for G/F/GS
and coupling is outside the A sweep unless separate parameterized RTL is added.
