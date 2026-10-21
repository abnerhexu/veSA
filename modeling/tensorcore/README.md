# Blackwell tensor arithmetic core

This directory contains an integer-only CPU reference and author-verified test
vectors for the standalone Chisel modules in `src/main/scala/gemmini/tensorcore`.
The implementation follows paper776 Algorithms 5, 7 and 9. Project-wide progress
is recorded in the parent repository under `docs/progress/`.

## Implemented scope

| A/B input | FP32 C/D | FP16 C/D | Fusion length |
| --- | --- | --- | --- |
| E4M3 (finite-only encoding) | Yes | Yes | 32 |
| E5M2 | Yes | Yes | 32 |
| FP16 | Yes | Yes | 16 |
| BF16 | Yes | No | 16 |
| MXFP4 E2M1, E8M0 scale / 32 elements | Yes | No | 64 |
| NVFP4 E2M1, UE4M3 scale / 16 elements | Yes | No | 64 |

A and B currently use the same format, selected at elaboration time. FP32 is
only an accumulator/output format. FP32, TF32 and FP64 operand paths do not
exist. Hopper is not implemented yet. The first 2x2 internal-SRAM Gemmini
RoCC/DMA path is verified; see the [P6 release scope and serial acceptance runner](../../integration/BLACKWELL_RELEASE.md).
FP4 uses the dedicated GST-FDPA path; it does
not implement the separate ST-FDPA instruction family or per-tensor NVFP4 alpha.

The new `TensorExecute` memory adapter loads raw A/B/C/scales and overwrites D
through banked row ports. See [local-memory.md](local-memory.md) for its protocol
and tests using the original Gemmini SRAM bank. This first P5 integration
layer does not by itself verify the full Controller/AccumulatorMem/DMA system.

The C/D bridge now uses the real shared `AccumulatorMemCore` (also inherited
by the original `AccumulatorMem` wrapper), with an accepted-request ownership
router and full-width response contract checks. Tests are available as
`scripts/test-tensorcore-memory.sh acc-PROFILE`, plus `contracts` and `router`.
The default-off `tensorcore=Some(profile)` configuration now wires a fenced
command frontend, engine and raw memory path into Controller. See
[commands.md](commands.md) for its four-command ABI, drain/visibility rules,
configuration helper, and `cmd-PROFILE`/`fence`/`isolation` regressions. Command
composition tests use the production banks. The separate
[real integration build](../../integration/README.md) compiles all Gemmini and
locked upstream sources and has passed all nine profiles through real RoCC,
MVIN and MVOUT, plus legacy OS/counter/reset checks. This does not claim CPU
boot, VM, 4x4 or full single-port RoCC/DMA coverage; see the recorded scope.

`TensorPE` is retained as a standalone scalar arithmetic fixture. It accepts a
request containing raw C bits and a block length in
`1..fusionLength` for unscaled formats, or exactly 64 for FP4. It accepts that
many raw A/B pairs. For unscaled inputs it buffers unnormalized exact
products, finds the common exponent including C, truncates each aligned
magnitude toward zero at F=25, and exactly sums before the final conversion.
Inputs advance on `valid && ready`; output is held until consumed. Reset aborts
the in-flight block. The production mesh does not instantiate this collector;
its physical PE is `TensorWSPE` below.

For FP4, `GroupedDotFrontend` computes exact 16-element group dots in quarter
units, multiplies the scale significands without normalization, and supplies
four terms to the same `FusedAccumulator` used by unscaled formats. The four
terms and C are aligned at F=35, exactly summed in 50 signed bits, and converted
to FP32 with RZ. The term magnitude has 23 fractional bits and 36 total bits,
enough for the larger scaled group sums. No bits are dropped before alignment.

Select `BlackwellConfig.mxfp4` or `BlackwellConfig.nvfp4`. Their operand bundle
adds `alpha` and `beta` scale bytes. MXFP4 scales stay constant across 32 element
transactions (two computation groups); NVFP4 scales stay constant across 16.
These contracts are asserted in simulation. NVFP4's ignored high byte bit is
masked before scale decoding or checking the block-constant contract.

`TensorMesh` produces a 2x2 output tile by default, but its compute fabric is a
weight-stationary `fusionLength x columns` array. Each `TensorWSPE` contains two
stationary B/scale banks, a 9-bit signed exponent lane and one exact
significand multiplier plus fixed-point adder. A is transposed into two buffers
at the left edge. It moves horizontally through the same PE rows twice; the
first wave performs only `(max,+)` exponent work, while the second wave aligns
products to the fixed maximum and carries a fixed-point sum vertically down
each column. No PE retains products.

The exponent result returns through a `fusionLength`-deep narrow FIFO. Nine
signed bits cover the implemented exponent range from the GST zero exponent
-139 through the largest scaled-group exponent 254. C is
decoded and aligned at the column top when that result returns. A second segment
may load and run its exponent wave while the first segment is replayed; a
dedicated regression checks uninterrupted bank-to-bank input and the overlap
latency bound. For K greater than the fusion length, the column-bottom result is
normalized to the configured FP32/FP16 format and becomes C for the next
segment. Thus a physical segment implements FDA and repeated segments implement
the ordered CoFDA chain without increasing physical depth with runtime K.

The request supplies C, K (1..65535) and a 16-bit tag; each input transaction
supplies one A column and one B row for loading. Input bubbles affect collection
only. The mesh holds the final matrix/tag until accepted and permits one matrix
request in flight.

For FP4, the physical WS column is 64 PEs deep and has exact group boundaries
after every 16 PEs. Each group carries an exact quarter-unit sum before its
scale generates one GST term; the four terms and C then use the same maximum
exponent and F=35 fixed-point sum. Mesh K must be a positive multiple of 64;
other lengths are rejected
by an assertion, with no implicit padding. Its optional `in.alpha` vector
supplies one scale byte per A row and `in.beta` one per B column for the current
K slice. Scale metadata moves with the corresponding operands through the
same skew/forwarding registers. The final FP32 result of each 64-element block
becomes C for the next block. A 32-element scale boundary does not round the sum.

## Reproducible tools and sequential tests

The standalone build uses Mill 1.3.0, JDK 17, Scala 2.13.10, Chisel 3.6.0,
chiseltest 0.6.0 and ScalaTest 3.2.15. It excludes Rocket/Chipyard-dependent
sources. Existing `build.sbt` is unchanged. JDK 17 is selected for compatibility
with the existing Scala compiler. The physical WS mesh regressions use
Verilator because the 16/32/64-deep arrays are too slow under Treadle; no GPU is
required.

Run commands from the Gemmini submodule (`veSA/`). Run only one test task at a
time: `-j 1` limits Mill task concurrency, and the test JVM heap is capped at
1536 MiB with one active processor (also limiting native simulator compilation).

```sh
mill --no-server -j 1 tensorcore.test.testOnly gemmini.tensorcore.BuildSmokeSpec
mill --no-server -j 1 tensorcore.test.testOnly gemmini.tensorcore.TensorPESpec
TENSORCORE_SIM=verilator mill --no-server -j 1 tensorcore.test.testOnly gemmini.tensorcore.TensorMeshSpec
TENSORCORE_SIM=verilator mill --no-server -j 1 tensorcore.test.testOnly gemmini.tensorcore.TensorWSMeshSpec
mill --no-server -j 1 tensorcore.test.testOnly gemmini.tensorcore.TensorWSStructureSpec
mill --no-server -j 1 tensorcore.test.testOnly gemmini.tensorcore.GroupedFrontendSpec
mill --no-server -j 1 tensorcore.test.testOnly gemmini.tensorcore.GroupedPESpec -- -z mxfp4
mill --no-server -j 1 tensorcore.test.testOnly gemmini.tensorcore.GroupedPESpec -- -z nvfp4
TENSORCORE_SIM=verilator mill --no-server -j 1 tensorcore.test.testOnly gemmini.tensorcore.GroupedMeshSpec
```

To run one format/output combination, append `-- -z e4m3-f32` to the PE or mesh
test command. Other labels are `e4m3-f16`, `e5m2-f32`, `e5m2-f16`, `fp16-f32`,
`fp16-f16`, and `bf16-f32`.

Generate the default 2x2 FP8 array as SystemVerilog:

```sh
mill --no-server -j 1 tensorcore.runMain gemmini.tensorcore.Generate e4m3 f32 out/rtl
mill --no-server -j 1 tensorcore.runMain gemmini.tensorcore.Generate mxfp4 f32 out/rtl-mxfp4
mill --no-server -j 1 tensorcore.runMain gemmini.tensorcore.Generate nvfp4 f32 out/rtl-nvfp4
```

The pure integer model has no third-party dependencies:

```sh
cd modeling/tensorcore
../../../.venv/bin/python -m unittest -v test_reference.py
```

For author-model differential validation, the tested environment uses
`mmasim==2.0.3` and `torch==2.9.0+cpu`. Install CPU torch in the parent project's
uv environment, then mmasim and its declared helper dependency:

```sh
uv pip install 'torch==2.9.0+cpu' --index-url https://download.pytorch.org/whl/cpu
uv pip install 'mmasim==2.0.3' ninja
```

Run validation from the Gemmini submodule. It imports the FDPA arithmetic
module directly, so no FMA C++ extension or CUDA kernels are built. CPU torch
threads are limited to one; batches are bounded to 256 cases.

```sh
../.venv/bin/python modeling/tensorcore/validate.py --cases 150000
../.venv/bin/python modeling/tensorcore/validate.py --cases 1000 --vectors modeling/tensorcore/vectors.tsv --matrix-vectors modeling/tensorcore/matrices.tsv
../.venv/bin/python modeling/tensorcore/validate_grouped.py --cases 1000
../.venv/bin/python modeling/tensorcore/validate_grouped.py --cases 500000
../.venv/bin/python modeling/tensorcore/validate_grouped.py --cases 1000 --vectors modeling/tensorcore/grouped-vectors.tsv --matrix-vectors modeling/tensorcore/grouped-matrices.tsv
```

The first command checks 1,050,000 scalar dot products across the seven legal
unscaled format/output combinations; the grouped 500000-case command checks
1,000,000 dots across MXFP4 and NVFP4. These are requested test counts, not
statements that the commands have already passed; actual outcomes are in
`docs/progress/`.
Vector contents use deterministic seeds per configuration and are independent
of the requested case count. Matrices include K=1,17,32,33,64,65 with both finite
moderate values and arbitrary raw bits. RTL tests also exercise bubbles,
backpressure and consecutive requests.

The grouped fixtures include 884 PE vectors (570 MXFP4, 314 NVFP4), all 256 FP4
operand encoding pairs, all 256 scale storage codes for both scale formats,
group cancellation, empty groups, subnormal C, infinities and canonical NaNs.
There are 12 grouped matrix vectors with K=64,128,192, independent row/column
scales, nonzero C and both finite and raw-bit cases. The group frontend test
checks downstream backpressure directly; PE tests also reset a partially filled
group and verify the next request is clean.

## Details taken from the pinned author model

MMA-Sim 2.0.3 `arithmetic/fdpa.py` and `nv_ptx/sim.py` supplement the paper:

- Input subnormals retain their format's minimum normal exponent; their
  significands do not acquire an implicit leading one.
- In T-FDPA, every zero term uses exponent -133 for FP32 output or -22 for Blackwell FP16
  output when selecting the alignment exponent, including zero C.
- RZ-FP32 normalization returns infinity on overflow, as the author model does;
  it does not saturate to the largest finite value as a generic IEEE RZ helper
  might. Subnormal output is retained, with signed underflow to zero.
- NaNs are canonicalized to `0x7fffffff` or `0x7fff`; invalid products and
  opposite-signed infinities produce the same canonical NaN.
- GST-FDPA uses zero exponent -139. A group that cancels to zero keeps its
  scale exponent; only groups with all original products zero (or zero scale)
  use the zero exponent. This distinction is covered by a directed CPU test.

The local model and RTL are independent implementations of these semantics.
The TSV fixtures are generated numerical data checked against the pinned author
model. Passing them establishes agreement with that reference; it does not
establish a local comparison with real NVIDIA hardware.
