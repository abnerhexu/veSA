# Hopper weight-stationary behavioral backend

The current `tensorcore/hopper` branch starts at the accepted Blackwell WS v2
commit `f197fdf`. Blackwell's branch and immutable release specification remain
available. P9 CPU, core and full-system verification has passed; historical OS
Hopper results belong to `tensorcore/hopper-os-v1` and are not reused here.
This is a paper/model-aligned systolic backend, not NVIDIA instruction or
timing compatibility and not a local GPU differential test.

## Numerical profile

`HopperConfig` implements the pinned MMA-Sim 2.0.3 **WGMMA/HGMMA** family.
It has seven static same-format A/B profiles:

| A/B | C/D | Fusion | Alignment F | Block-end conversion |
| --- | --- | --- | --- | --- |
| E4M3 / E5M2 | FP32 | 32 | 13 | RZ-E8M13 in FP32 storage |
| E4M3 / E5M2 | FP16 | 32 | 13 | RNE-FP16 |
| FP16 / BF16 | FP32 | 16 | 25 | RZ-FP32 |
| FP16 | FP16 | 16 | 25 | RNE-FP16 |

Both reduced internal precision and block-end conversion are necessary for
FP8/FP32. The next fusion block sees the rounded D, not a wider hidden sum.
RZ-E8M13 uses the normal FP32 exponent and clears the low ten fraction bits of
finite outputs, including subnormals. Canonical NaNs remain `0x7fffffff`.
The implementation truncates **magnitudes** before restoring signs, so negative
terms round toward zero rather than toward negative infinity.

Instruction-family clarification: paper776 Table 6 specifies F/L/rho but not
zero-term exponents. The pinned `nv_ptx/sim.py` WGMMA wrapper uses `e_zero=-133`
for every profile, including half output; its older Hopper MMA wrapper instead
uses -21 for half output. Blackwell uses -22. This branch explicitly selects
the WGMMA family, not all Hopper instruction families. The FP16/FP16 directed
case `[0x0800,1] * [0x0c00,1] + 0` rounds to half subnormal `0x0001` here but
`0x0000` with Blackwell's zero exponent. F=25 and RNE-FP16 are unchanged; the
otherwise unspecified zero handling is not silently treated as identical.

No FP32/TF32/FP64 A/B, BF16 output, mixed A/B format, or native FP4/scales
profile is added. `HopperConfig(E2M1)` and `hopper-mxfp4`/`hopper-nvfp4`
generation requests are rejected. Blackwell MXFP4/NVFP4 remain separately
selectable with `BlackwellConfig`, never relabeled as native Hopper.

## Gemmini integration

`TensorConfig` is the shared static interface. The original `BlackwellConfig`
API and defaults retain their numeric behavior. `TensorCoreConfigs.hopper()`
and `HopperTensorCoreConfig` select the new backend; the four-command private
RV64 ABI, raw memory layout, single-task fence and write visibility rules
are unchanged. Header output now identifies architecture, internal F and
output fraction precision.

Both architectures use the same physical WS topology. The logical 2x2 output
tile is backed by `fusionLength x 2` PEs: 32x2 for Hopper FP8 and 16x2 for
FP16/BF16. Each PE has two stationary weight banks, one input-product lane and
a 9-bit exponent lane. The exponent wave and significand wave share the array;
the next segment's exponent wave can overlap the current segment's replay.

The first validated system target remains 2x2, RV64 machine mode, internal
SRAM, single-port scratchpad, dual-port accumulator, no clock gating and a
64-bit TileLink beat. Factories are not claims of extra supported systems.
See [Blackwell scope](BLACKWELL_RELEASE.md) for the shared exclusions (CPU/VM
boot, external SRAM, packed FP4, performance, etc.).

## Serial verification

Use the parent uv environment; run exactly one task at a time. Dependency
setup remains `scripts/setup-tensorcore-integration.sh`. The generic runner
records input source hashes, subprocess logs/exit codes and exact ScalaTest
counts under ignored `out/hopper-verification/STAGE-*/`. It checks that inputs
do not change during a stage, retains the original Blackwell fixtures and
checks the pinned author code/package identities. It does not weaken the
original Blackwell release verifier's source freeze.

```sh
../.venv/bin/python scripts/verify-hopper.py cpu
../.venv/bin/python scripts/verify-hopper.py core
../.venv/bin/python scripts/verify-hopper.py integration
../.venv/bin/python scripts/verify-hopper.py blackwell
```

`cpu` runs directed integer-model tests, 1,050,000 Hopper dots, 1,050,000
Blackwell unscaled dots and 1,000,000 Blackwell grouped dots. `core` checks
profile guards, Hopper PE/WS mesh, physical lane/PE structure, and every
Blackwell PE/WS mesh/grouped profile.
`integration` compiles all real Gemmini/upstream sources, emits Hopper FP8
and FP16 top-level RTL and runs the seven actual RoCC/DMA profiles.
`blackwell` reruns all nine original real RoCC/DMA profiles in this branch.
JVM heaps remain 1536 MiB, one active processor, Mill `--no-server -j 1` and
native builds `-j1`. Release runners share an exclusive lock; do not launch
manual tests concurrently.

For a single task:

```sh
./scripts/test-tensorcore-hopper.sh config
./scripts/test-tensorcore-hopper.sh pe-e4m3-f32
./scripts/test-tensorcore-hopper.sh mesh-e4m3-f32
./scripts/test-tensorcore-hopper.sh integration-e4m3-f32
./scripts/test-tensorcore-hopper.sh rtl-e4m3
mill --no-server -j 1 tensorcore.runMain gemmini.tensorcore.Generate hopper-e4m3 f32 out/hopper-mesh
```

`validate.py --architecture hopper` generates separate `hopper-vectors.tsv`
and `hopper-matrices.tsv`; it never replaces Blackwell golden vectors. The
Hopper fixtures include C-only normal/subnormal precision, sign/cancellation,
block-output rounding after sum growth, and K=33 cancellation that detects
incorrectly postponing rounding until the very end. The CPU model uses integer
bit operations and compares against the pinned author's FDPA implementation,
not against RTL output or a host floating-point FMA.

All four P9 stages passed freshly on this WS branch: 3,100,000 random CPU
cases, 40 core tests, seven Hopper system configurations (88 matrices) and
nine Blackwell system configurations (96 matrices). The core gate covers 430
Hopper PE vectors, 88 Hopper matrices, 1267 Blackwell PE vectors and 96
Blackwell matrices, plus both architectures' double-bank overlap and physical
lane/PE structure. Both system suites retain E4M3 command-control scenarios.
The passing reports and their SHA-256 values are recorded in the parent P9
progress record; the earlier missing-dependency run is retained as a failure.

Pinned Hopper fixture SHA-256:

- `hopper-vectors.tsv`: `0e8abe39a9e6c3070773393c201d3ad282da2cea97fbf5ed549704268d814ab9`
- `hopper-matrices.tsv`: `cd94b71cc1f5e482f3d9cb87522dc31fd66f25bdcea5303bc28c9162c9292926`

Passing one stage alone does not reproduce this acceptance conclusion. Full
results, failures and excluded configurations are recorded in the progress
record. This conclusion is limited to the declared paper/author-model scope;
no real NVIDIA GPU differential test was performed.
