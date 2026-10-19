# Local-memory execution adapter (P5, first integration layer)

`TensorExecute` connects `TensorMesh` to banked row memories. It loads raw C,
streams A/B and scales, and overwrites raw D. This is an execution adapter,
not a RoCC instruction decoder. `TensorAccumulatorBridge` and the bank router
now connect the raw C/D protocol to the shared production accumulator storage
core. The original `Controller` now optionally connects a fenced command path;
see [commands.md](commands.md). The adapter tests described here do not verify
the full reservation-station/DMA system or complete P5/P6.

## Storage protocol

Geometry is an elaboration-time `TensorMemoryConfig`: default square 2x2 mesh,
four 256-row scratchpad banks, two 16-row accumulator banks, and two cycles of
write commit latency. Banks/depths are powers of two; one bank is allowed.
Bank selection uses high flat row-address bits, as in Gemmini's `LocalAddr`.
Requests contain plain row indices, without `LocalAddr` control/garbage bits.
Out-of-range requests and illegal K values assert at the direct adapter port;
the command frontend validates them first and returns status without launching.

| Descriptor field | Layout |
| --- | --- |
| `a` | K rows; row t contains A[0,t], A[1,t], ... (explicitly pretransposed) |
| `b` | K rows; row t contains B[t,0], B[t,1], ... |
| `c` | dim accumulator rows, row-major C; ignored when `zeroC` is true |
| `d` | dim accumulator rows, row-major D; may equal or overlap C |
| `alpha` / `beta` | FP4 only; K/scaleBlockSize rows of per-lane scale bytes |
| `k` | 1..65535 for unscaled formats; positive multiple of 64 for FP4 |
| `tag` | Opaque 16-bit completion tag |

Lane zero occupies the least significant bits. Scratchpad slots are the input
format width, except FP4 which uses a full byte and ignores the upper nibble.
Scales occupy one byte per slot in FP4 profiles. Accumulator slots are always
32 bits: FP32 C/D are untouched raw bits; FP16 C is the low 16 bits (upper bits
ignored) and FP16 D is zero-extended to 32 bits. This storage convention does
not numerically convert FP16 C to FP32 or FP32 C to FP16.

All initial C rows are fetched before compute. The mesh rounds at its own
fusion boundaries, independent of row/bank boundaries. Scale rows are fetched
once per shared-scale block and retained between K slices. A and B may be in
the same bank: the adapter serializes reads and tracks only one outstanding
read, so response tags or fixed latency assumptions are unnecessary.

## Ports and ownership

- `cmd` and `completed` use ready/valid. Only one command is in flight;
  `completed` holds the tag, and `busy` stays high, until completion is accepted.
- `spad` uses the existing `gemmini.ScratchpadReadIO` per bank. No DMA-tagged
  responses may be routed to the adapter.
- `accRead` uses the same raw-row read protocol with `dim*32` data bits. Its
  backend MUST supply original full-width words, bypassing numeric conversion,
  scaling, activation and normalization. Do not attach the legacy narrowed
  execution response blindly.
- `accWrite` uses ready/valid, byte masks and `accumulate=false`. A Gemmini
  bridge must drive the accumulator write `acc` control false; adding D to
  the old memory contents would incorrectly include C twice.
- Completion follows the last accepted write by `writeLatency` full cycles.
  That latency must cover the backend's actual write visibility delay, not
  just an enqueue delay in an unbounded external-memory queue. The fenced
  command frontend additionally waits for actual production-bank idle before
  reporting completion; its single-port tests deliberately rely on that check.
- The enclosing controller must fence legacy execution/DMA access during
  `busy` and wait for older writes before accepting a command. No arbitration
  or reservation-station hazard tracking is silently assumed here.
- Reset aborts control state. The enclosing system must also reset/drain
  outstanding memory responses; already accepted D writes are not rolled back.

`ScratchpadBank`, its row IO, and `ExtMemIO` have been moved without semantic
changes into `src/main/scala/gemmini/memory/` (still package `gemmini`). Both
the original Gemmini sources and the standalone build use these same classes.
`TensorExecuteSpec` retains the raw-SRAM baseline. `TensorAccumulatorExecuteSpec`
uses the real production accumulator storage core for C/D, as described below.

## Sequential verification

Run from the submodule. Each command runs one configuration; do not start two
tasks at once. Test heap remains limited to 1536 MiB.

```sh
mill --no-server -j 1 tensorcore.test.testOnly gemmini.tensorcore.TensorExecuteSpec -- -z e4m3-f32
```

The default simulator is Treadle. If a working `verilator_bin` and its matching
installed include/make files are available, the memory tests also support a
compiled backend without depending on Perl's Verilator launcher:

```sh
./scripts/test-tensorcore-memory.sh e4m3-f32
./scripts/test-tensorcore-memory.sh control
```

`TENSORCORE_VERILATOR_BIN` can specify the binary's absolute path. Otherwise
`verilator_bin` is resolved on PATH, with its compiled-in `VERILATOR_ROOT`.
Mill explicitly prepends the project launcher to the test subprocess PATH;
the script selects the native backend and accepts only a single listed profile.
The test JVM exposes only one processor to keep native build concurrency one.
This launcher forwards normal compile/version flags only; it is not a general
replacement for the upstream debug/hierarchical-build launcher.
For this host's Verilator 5.052, the generated Chiseltest 0.6 C++ harness needs
the `WData` to `EData` word-type compatibility define in `TensorTestBackend`.
This affects the test harness, not RTL arithmetic. The native tests use 2x2
geometry (no externally visible signal wider than 64 bits).

Replace the filter with `e4m3-f16`, `e5m2-f32`, `e5m2-f16`, `fp16-f32`,
`fp16-f16`, `bf16-f32`, `mxfp4` or `nvfp4` for the other configurations.
The tests use all 84 unscaled and 12 grouped author-verified matrix fixtures,
plus a zero-C command per profile. They cover same-bank A/B, bank crossing,
in-place C/D, nonzero C, dirty unused storage bits, output overwrite, adjacent
row sentinels, scale fetch counts, delayed requests/responses, write stalls,
completion stalls and consecutive commands.

`TensorExecuteControlSpec` additionally checks a single-bank geometry and a
joint controller/SRAM reset before any D write, followed by a fresh command.

Generate the adapter's standalone SystemVerilog:

```sh
mill --no-server -j 1 tensorcore.runMain gemmini.tensorcore.GenerateExecute e4m3 f32 out/execute-e4m3
mill --no-server -j 1 tensorcore.runMain gemmini.tensorcore.GenerateExecute nvfp4 f32 out/execute-nvfp4
```

Actual results, including anything not run, are recorded in the parent
repository's `docs/progress/`, not inferred from the commands above.

## Production accumulator bridge and ownership routing

The original `AccumulatorMem` now inherits `AccumulatorMemCore`, with its
constructor and implicit Arithmetic API retained. The independently compiled
core contains the original SRAM, write pipelines, masks, read queues, RMW
interface and single-port sub-bank queues. It does not require HardFloat or a
fake Arithmetic implementation: the shared adder was already external.
`AccumulatorIO.scala` contains the unchanged data protocol without unnecessary
Arithmetic type bounds. The original `SyncMem.scala` is compiled directly.

`TensorAccumulatorBridge` maps raw C/D to `AccumulatorBankClientIO`. It drives
full-width unconverted reads, zero arithmetic metadata and `acc=false` writes.
`AccumulatorBankRouter` then routes raw and legacy requests to the actual bank.
Its accepted-request owner FIFO is necessary because `fromDMA=false` identifies
both legacy execute and raw Tensor reads. Read response stalls preserve the
owner; queue exhaustion stalls new requests. Same-row read/write requests are
serialized, and reset prevents new handshakes. Reset must also reset/drain the
backend response queues; already committed writes are not rolled back.

The original `Scratchpad` has an optional `tensorcore_acc` bank-level port,
enabled by `GemminiArrayConfig.tensorcore_raw_access` (default false). It uses
this same router before normalization/scaling and preserves the legacy route.
The option currently requires 8/16-bit storage slots, 32-bit accumulator slots
and internal storage. Raw access alone leaves the Controller ports idle and
does NOT enable a new RoCC instruction. `tensorcore=Some(profile)` additionally
wires the fenced command/engine path; see [commands.md](commands.md). Per-port
arbitration does NOT replace its matrix-level DMA/legacy fence. Full system
verification remains separate from these production-bank tests.

`AccumulatorScaleResp` now checks that full_data slots are not narrower than
converted data slots. The reversed type arguments in the original Scratchpad
and ExecuteController were fixed with named arguments. Raw C is read directly
at the bank, rather than relying on the scale/normalizer response.

Run one configuration at a time:

```sh
./scripts/test-tensorcore-memory.sh contracts
./scripts/test-tensorcore-memory.sh router
./scripts/test-tensorcore-memory.sh acc-e4m3-f32
./scripts/test-tensorcore-memory.sh acc-nvfp4
./scripts/test-tensorcore-memory.sh acc-single-mxfp4
```

All nine profiles support the `acc-` prefix. The single-port runner supports
`acc-single-e4m3-f32`, `acc-single-mxfp4`, `acc-single-nvfp4`. It uses two sub-banks
and a four-cycle completion visibility delay, instead of two for dual-port
storage. Independent contract tests also cover one sub-bank, overwrite byte
masks and actual UInt RMW addition; Tensor execution poisons the adder result
and asserts that its valid signal stays false.

The storage core also fixes two pre-existing internal single-port elaboration
bugs: accessing absent ext_mem and selecting a negative-width single-sub-bank
index. External/shared memory variants and the full legacy normalizer/scaler
system are not claimed verified by these tests.

```sh
mill --no-server -j 1 integrationSyntax.compile
```

This last target ONLY parses edited integration sources. It does not typecheck
Rocket/Chipyard/HardFloat or replace full system elaboration.

Export the typed bridge, ownership router or the actual storage core:

```sh
mill --no-server -j 1 tensorcore.runMain gemmini.tensorcore.GenerateAccumulator bridge out/accumulator-bridge
mill --no-server -j 1 tensorcore.runMain gemmini.tensorcore.GenerateAccumulator router out/accumulator-router
mill --no-server -j 1 tensorcore.runMain gemmini.tensorcore.GenerateAccumulator dual out/accumulator-dual
mill --no-server -j 1 tensorcore.runMain gemmini.tensorcore.GenerateAccumulator single out/accumulator-single
```

These 32-bit ports carry raw C/D storage words, not FP32 A/B MMA operands.
