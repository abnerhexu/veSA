# Blackwell tensor arithmetic core

This directory contains an integer-only CPU reference and author-verified test
vectors for the standalone Chisel modules in `src/main/scala/gemmini/tensorcore`.
The implementation follows paper776 Algorithms 5 and 7. Project-wide progress
is recorded in the parent repository under `docs/progress/`.

## Implemented scope

| A/B input | FP32 C/D | FP16 C/D | Fusion length |
| --- | --- | --- | --- |
| E4M3 (finite-only encoding) | Yes | Yes | 32 |
| E5M2 | Yes | Yes | 32 |
| FP16 | Yes | Yes | 16 |
| BF16 | Yes | No | 16 |

A and B currently use the same format, selected at elaboration time. FP32 is
only an accumulator/output format. FP32, TF32 and FP64 operand paths do not
exist. FP4 RTL, block-scale hardware, Hopper and the existing Gemmini
execution/memory controller integration are not implemented yet. A separate
integer GST-FDPA reference (`grouped_reference.py`) is available for the next
FP4 hardware stage.

`TensorPE` accepts a request containing raw C bits and a block length in
`1..fusionLength`, then that many raw A/B pairs. It buffers unnormalized exact
products, finds the common exponent including C, truncates each aligned
magnitude toward zero at F=25, and exactly sums before the final conversion.
Inputs advance on `valid && ready`; output is held until consumed. Reset aborts
the in-flight block.

`TensorMesh` is an output-stationary 2x2 array by default, parameterized by row
and column count. A moves horizontally and B vertically through registers. Its
request supplies C, K (1..65535) and a 16-bit tag; each input transaction supplies
one A column and one B row. It performs internal wavefront skewing and stalls
forwarding on input bubbles. It rounds at the specified fusion boundaries,
reuses the rounded result as the next C, and holds the final matrix/tag until
accepted. Only one matrix request is in flight.

## Reproducible tools and sequential tests

The standalone build uses Mill 1.3.0, JDK 17, Scala 2.13.10, Chisel 3.6.0,
chiseltest 0.6.0 and ScalaTest 3.2.15. It excludes Rocket/Chipyard-dependent
sources. Existing `build.sbt` is unchanged. JDK 17 is selected for compatibility
with the existing Scala compiler. Treadle simulation requires no GPU or Verilator.

Run commands from the Gemmini submodule (`veSA/`). Run only one test task at a
time: `-j 1` limits Mill task concurrency, and the test JVM heap is capped at
1536 MiB with two active processors.

```sh
mill --no-server -j 1 tensorcore.test.testOnly gemmini.tensorcore.BuildSmokeSpec
mill --no-server -j 1 tensorcore.test.testOnly gemmini.tensorcore.TensorPESpec
mill --no-server -j 1 tensorcore.test.testOnly gemmini.tensorcore.TensorMeshSpec
```

To run one format/output combination, append `-- -z e4m3-f32` to the PE or mesh
test command. Other labels are `e4m3-f16`, `e5m2-f32`, `e5m2-f16`, `fp16-f32`,
`fp16-f16`, and `bf16-f32`.

Generate the default 2x2 FP8 array as SystemVerilog:

```sh
mill --no-server -j 1 tensorcore.runMain gemmini.tensorcore.Generate e4m3 f32 out/rtl
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
```

The first command checks 1,050,000 scalar dot products across the seven legal
format/output combinations. This is a requested test count, not a statement
that the command has already passed; actual outcomes are in `docs/progress/`.
Vector contents use deterministic seeds per configuration and are independent
of the requested case count. Matrices include K=1,17,32,33,64,65 with both finite
moderate values and arbitrary raw bits. RTL tests also exercise bubbles,
backpressure and consecutive requests.

## Details taken from the pinned author model

MMA-Sim 2.0.3 `arithmetic/fdpa.py` and `nv_ptx/sim.py` supplement the paper:

- Input subnormals retain their format's minimum normal exponent; their
  significands do not acquire an implicit leading one.
- Every zero term uses exponent -133 for FP32 output or -22 for Blackwell FP16
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
