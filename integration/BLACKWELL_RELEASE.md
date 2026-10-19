# Blackwell v1 behavioral release

Blackwell v1 has passed the numerical/system acceptance gates for the exact
scope below. The final acceptance/release-commit record lives in the parent
repository at `docs/progress/2026-10-19-08-blackwell-release.md`. A successful
preflight or one profile alone is not sufficient to reproduce that conclusion.

The implemented arithmetic matches the two project papers and MMA-Sim 2.0.3
on the recorded bit-exact CPU/RTL tests. This is **not** a differential test
against a local NVIDIA GPU, NVIDIA instruction compatibility, or a claim about
NVIDIA timing, throughput or physical microarchitecture.

## Frozen first-release scope

The machine-readable specification and hashes are in
[blackwell-release.json](blackwell-release.json). The validated implementation
base is `d3a836c1ae0a273a40f941f5f1c8c93051ea306a`. P6 adds release tooling and
documentation without changing that implementation, build graph or tests.
Its final release commit is also recorded in the parent's acceptance
record/gitlink; the manifest does not attempt to embed its own commit hash.

| Static A/B profile | C and D | Fusion length | Aligned fractional bits | Block-end conversion |
| --- | --- | --- | --- | --- |
| E4M3 / E5M2 | FP32 | 32 | 25 | RZ-FP32 |
| E4M3 / E5M2 | FP16 | 32 | 25 | RNE-FP16 |
| FP16 / BF16 | FP32 | 16 | 25 | RZ-FP32 |
| FP16 | FP16 | 16 | 25 | RNE-FP16 |
| Dedicated MXFP4 E2M1, E8M0 scale / 32 | FP32 | 64 | 35 | RZ-FP32 |
| Dedicated NVFP4 E2M1, UE4M3 scale / 16 | FP32 | 64 | 35 | RZ-FP32 |

These are nine profiles. A and B share the same format and are selected at
elaboration. Each unscaled block has length `min(remaining K, fusion length)`;
the rounded D becomes the next block's C. FP4 requires a positive multiple of
64 and computes exact 16-element groups, without rounding at scale boundaries.
FP32 is an accumulation/output interface only, not an A/B multiply path.

The released integration is `TensorCoreConfigs.blackwell(profile)` with a
2x2 output-stationary array, RV64 machine mode, ungated clock, internal SRAM,
four single-port scratchpad banks of 256 rows and two dual-port accumulator
banks of 16 rows. One task owns the memory at a time. The tested system uses
64-bit TileLink beats and real Gemmini MVIN/MVOUT, DMA, ROB and controllers.
The CPU-facing interface is injected RoCC, not CPU instruction execution.

Operate on complete 2x2 output tiles; no partial-M/N masking or automatic
large-matrix scheduling is supplied. K is encoded in 16 bits and must also fit
all local address ranges. This is not a promise that K=65535 fits the small
SRAM configuration. Fixed integration vectors cover unscaled K=1/17/32/33/64/65
and FP4 K=64/128/192; legal K values are not exhaustively tested.

A is pretransposed/K-major in scratchpad, B is row-major; each row carries two
raw 8- or 16-bit operand slots. FP4 occupies a byte per element, with the high
nibble ignored. C/D occupy two raw 32-bit slots per accumulator row, including
FP16 C/D in the low halfword (D upper half zero). Scales occupy byte slots.
Disable legacy numeric DMA transforms. See [memory layout](../modeling/tensorcore/local-memory.md)
and the [four-command RV64 ABI](../modeling/tensorcore/commands.md).

The ABI uses funct 32/33/34/35 for AB/CD/scales/Run. It drains older work,
fences younger commands, overwrites D without a second FP add, and retains
ownership through actual write visibility and completion-response backpressure.
Use `xd=1` to observe rejection status. Reset aborts pending commands and
clears descriptors; it is a joint system reset, not rollback of committed
writes or protection from external agents.

Not released: Hopper, FP32/TF32/FP64 A/B, BF16 output, FP6, unscaled FP4,
generic ST-FDPA/MXFP8/MXFP6/MXFP4, packed FP4 storage, NVFP4 per-tensor FP32
alpha, 4x4 end-to-end operation, single-port accumulator end-to-end operation,
external/shared SRAM, clock gating, VM/PTW, CPU/Linux boot, full legacy ISA
regression, multi-task execution, CUDA/PTX/SASS/TMEM compatibility or performance
alignment. Some factory/component options exist; they do not expand this list.

## Locked inputs and semantics

The manifest pins the two paper hashes, four TSV fixture hashes (1267 PE
vectors and 96 matrices), local integer models/validators, and installed
MMA-Sim arithmetic/helper/instruction-wrapper source hashes. CPU dependencies
are `mmasim==2.0.3` and `torch==2.9.0+cpu`; no CUDA extension is loaded.
[dependencies.lock](dependencies.lock) pins the four actual upstream source
commits. The build pins Mill 1.3.0, JDK 17, Scala 2.13.10, Chisel 3.6.0,
chiseltest 0.6.0 and ScalaTest 3.2.15. The tested native backend is Verilator
5.052, with the checked-in compatibility wrapper/flags. The Mill launcher can
use a different JDK; target compilation/testing selects JDK 17 in `build.mill`.
The runner records Python/tool versions instead of assuming the shell's Java
version is the target compiler's version.

Known nonfatal warnings: the one-source host TileLink client produces four
upstream TLMonitor dynamic-index-width warnings. The locked upstream sources,
protocol monitors and assertions are retained; warnings are not hidden by
disabling verification.

Author-code supplements are part of this release: subnormal inputs are retained;
zero alignment exponents are -133 (FP32), -22 (FP16), -139 (GST); a cancelling
GST group retains its scale exponent; NVFP4 scale bit 7 is ignored; NaNs are
canonical `0x7fffffff`/`0x7fff`; the author's RZ-FP32 overflow returns infinity.
See the [arithmetic specification](../modeling/tensorcore/README.md#details-taken-from-the-pinned-author-model).
No paper/model conflict remains unresolved within the declared profile scope.

## Acceptance evidence

| Gate | Accepted evidence |
| --- | --- |
| Integer reference and pinned author | P6 rerun: 11 directed tests, 1,050,000 unscaled + 1,000,000 grouped dots |
| Frozen fixtures | P6 rerun: 7000 + 2000 differential dots, all four exported files byte-identical |
| Standalone PE / mesh / group frontend | P4 passed all nine profiles, 1267 PE vectors and 96 matrices; six arithmetic modules and five corresponding tests are unchanged from `0d2d027` |
| Production storage / command isolation | P6 rerun: 13 tests including real masked/RMW storage, ownership, drain/fence and reset |
| Real Gemmini system | P6 rerun: complete source compile, enabled/default-off RTL, all nine profiles and 96 matrices through real RoCC/MVIN/MVOUT |
| Release runner | P6 rerun: eight standard-library tests, including failed commands, wrong/zero test counts and exclusive locking |

The standalone core evidence commit in the table is
`0d2d027c195095ec13218e62a692a336b85a9c65`; see the parent progress record for
the complete source-identity check and historical results. P6 deliberately
reuses that unchanged-core evidence; it did not rerun the `core` or `all`
commands. This is distinct from the fresh P6 system and CPU runs above.

## Reproduce serially

From the Gemmini submodule, using the parent project's uv environment:

```sh
./scripts/setup-tensorcore-integration.sh
../.venv/bin/python scripts/verify-blackwell-release.py check
../.venv/bin/python -m unittest discover -s scripts -p test_blackwell_release.py -v
../.venv/bin/python scripts/verify-blackwell-release.py fixtures
../.venv/bin/python scripts/verify-blackwell-release.py cpu
../.venv/bin/python scripts/verify-blackwell-release.py core
../.venv/bin/python scripts/verify-blackwell-release.py memory
../.venv/bin/python scripts/verify-blackwell-release.py integration
```

Alternatively, `../.venv/bin/python scripts/verify-blackwell-release.py all`
executes those five verification stages sequentially after preflight (but not
the runner's separate self-tests or dependency installation). Use `--list` to
inspect commands without running them. `core` uses the original standalone
tests/backends; `memory` and real integration use Verilator. Every subprocess
finishes before the next starts; there is no resume/cache of PASS results.
Compilation itself can reuse Mill's valid incremental cache.

`fixtures` runs the 11 integer-model unit tests, checks 7000 unscaled and 2000
grouped random dots against the author model, regenerates all four TSV files
under a new ignored output directory and demands byte-for-byte hash equality.
It never replaces the checked-in golden vectors. `cpu` checks 1,050,000 unscaled
and 1,000,000 grouped dots (seeds 776/2899), in bounded single-threaded batches.
`core` covers PE/mesh/group-frontend handshakes and arithmetic for all nine
profiles. `memory` covers width contracts, masked legacy RMW, accepted-read
ownership, late writes, drain/fence, reset and isolation. `integration` performs
full real-source compilation, enabled/default-off RTL generation, and nine
separate real RoCC/DMA profile simulations, including E4M3 control scenarios.
Optional per-profile/component commands remain documented in the linked READMEs.

The runner holds a nonblocking file lock against another release runner, uses
Mill `--no-server -j 1`, JVM heaps of 1536 MiB, one active processor and native
`MAKEFLAGS=-j1`. **Do not run manual tests or another simulator concurrently**;
the lock cannot stop commands launched outside this runner. Existing CPU/GPU
training or unrelated builds are not automatically controlled.

Each invocation leaves a unique `out/blackwell-release/STAGE-*/` directory with
`result.json`, individual command logs, command/elapsed-time/exit-code records,
log hashes, pre/post source identity and tool information. Failed commands stop
the stage, retain their log, and write a failed result. A ScalaTest exit code of
zero is insufficient: the exact expected test count and `All tests passed.`
summary are required, so a typo filtering out every test cannot pass.
All inputs are rechecked after a run. The runner refuses altered production
sources, fixtures, author files or upstream checkouts; docs-only changes are
allowed. It does not install dependencies, fetch, commit, tag, push or branch.

An individual `passed` result means that stage passed, not all of P6. The
acceptance record explicitly distinguishes fresh runs from earlier evidence
whose source/vector identity was preserved. Raw logs/generated RTL stay ignored;
the parent progress record preserves the reviewable outcome and release commit.

## Blackwell-to-Hopper handoff

P6 must pass and have a fixed Blackwell release commit before creating
`tensorcore/hopper`. Keep the Blackwell branch and fixtures reproducible.
Hopper FP8 changes both F=13 and block-end RZ-E8M13 for FP32 output; changing
only F is insufficient. FP16/BF16 keep the specified behavior, and Hopper must
reject native MXFP4/NVFP4. No Hopper implementation is part of this release.
