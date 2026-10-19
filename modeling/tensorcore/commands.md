# Fenced Blackwell command path

This is a private RV64 Gemmini ABI, not NVIDIA PTX/CUDA compatibility. The
arithmetic profile is selected at elaboration, not by reinterpreting a command
at run time. The command/queue/engine/production-bank composition is tested
independently. A separate [real integration target](../../integration/README.md)
now compiles all Gemmini sources and connects its actual RoCC/DMA path to a
TileLink RAM; see the progress record for the profiles actually run.

## Configuration

`GemminiArrayConfig.tensorcore: Option[BlackwellConfig]` defaults to `None`.
`Some(profile)` enables the command frontend, engine, raw C/D bridge and bank
router in the original Controller/Scratchpad. It automatically enables the
raw ports; `tensorcore_raw_access=true` alone still leaves them tied idle.

The first integrated configuration requires RV64, an ungated clock, internal
non-dummy SRAM, 32-bit accumulator slots, and operand storage slots of exactly
8 or 16 bits, matching the selected profile. Bank counts/depths must meet
`TensorMemoryConfig`'s power-of-two constraints. The physical array is square.
No FP32/TF32/FP64 A/B path is introduced.

`TensorCoreConfigs.blackwell(profile, dim=2, singlePortedAccumulator=false)`
provides a small configuration with four 256-row scratchpad banks, two 16-row
accumulator banks, and no legacy DMA numeric scaling or normalization.
`BlackwellTensorCoreConfig` is the corresponding Chipyard mixin. A 4x4 factory
option exists, but this batch's command regression uses 2x2, not 4x4 or a full
SoC. Do not treat the mixin's parse check as proof of system elaboration.

For an existing configuration, select `tensorcore=Some(profile)` explicitly.
Disable mvin/mvout numeric transforms when transporting encoded data: even
floating point scale 1.0 is not a bit-preserving transformation of raw integer
carriers. Use full-width raw accumulator loads/stores for C/D, including FP16
C/D (one halfword in each 32-bit slot). Header generation exports enabled,
profile, output width, scale-block and command constants. A/B FP4 uses one
byte per element, not packed nibbles.

## Encoding

Use the configured Gemmini custom opcode. `funct` is seven bits. Addresses are
**flat row indices**, `bank * bankDepth + row`; they are neither byte pointers
nor `LocalAddr` values with metadata bits. A/B and scale rows are scratchpad
addresses; C/D are accumulator addresses. See [local-memory.md](local-memory.md)
for lane packing and the required K-major/pretransposed A layout.

| funct | Operation | rs1 | rs2 |
| --- | --- | --- | --- |
| 32 | Set A/B | A row, low 32 bits | B row, low 32 bits |
| 33 | Set C/D | C row, low 32 bits | D row, low 32 bits |
| 34 | Set scales (FP4 only) | Alpha row, low 32 bits | Beta row, low 32 bits |
| 35 | Run one full tile | `[15:0] K`, `[31:16] tag`, `[32] zeroC` | zero |

All unspecified bits must be zero. K must be positive; FP4 requires a multiple
of 64. `zeroC=1` skips the C read and its address validation; D is still checked.
AB/CD descriptors must have been set since reset, as must scales for FP4.
Descriptors persist across runs. Bad encodings never partially update them.
Address bounds, including overflow carry and the whole C/D tile, are checked
before the engine can issue any memory request. Scale ranges use the selected
16- or 32-element sharing block, not the 64-element fusion length.

With `xd=1`, every command returns one response to its captured `rd`:

- `[31:16]`: run tag, or zero for descriptor commands.
- `[15:8]` and all bits above bit 31: zero.
- `[7:0]`: 0 success, 1 missing descriptor, 2 bad encoding/K, 3 bad range.

Use `xd=1` to observe errors. With `xd=0`, no response is manufactured; younger
commands are still blocked until the task and its writes have drained.
Invalid runs do not enter the engine or modify memory. Descriptor commands
validate encoding immediately; their address ranges are validated at Run.

## Ordering and memory visibility

1. The original two-entry input queue feeds `TensorCommandGate` before either
   loop unroller. A Tensor command is captured by the frontend, not the ROB.
2. It blocks **dispatch** of all younger commands, including descriptor and
   counter commands. Two younger entries may still be buffered. Older loop,
   ROB, load/store/execute and counter responses continue to drain. The blocked
   queue head is deliberately excluded from the older-work test.
3. Drain includes accepted DMA loads through their final local response
   (scaler, zero-fill and repeater included), pending local read responses,
   DMA store queues, and the real accumulator write pipeline/sub-bank queues.
   `MemoryActivity` tracks responses; `AccumulatorMemCore.io.busy` exposes
   physical queued work. Stream-reader idle alone is insufficient.
4. Only then does exclusive ownership select the Tensor scratchpad read path.
   Legacy/DMA request ports are gated, and invariants assert they remain idle.
   Raw C bypasses normalization/scaling; D always overwrites with `acc=false`.
5. Engine completion is followed by an actual memory-idle check. A guessed
   write delay cannot release a pending sub-bank write. Ownership remains
   exclusive while a successful response is backpressured; younger work is
   released only after response acceptance (or implicit completion for xd=0).

Counter and Tensor CPU responses share an arbiter. Descriptor commands also
fence older work, keeping the first version conservative and easy to verify.
This is command-order isolation, not multi-process protection or a promise of
concurrent tasks. Shared/external SRAM and clock gating are out of scope here.

Reset aborts the frontend/engine and clears descriptors and ownership tracking.
It must be shared with the memory-side queues/clients; it cannot roll back
already committed writes. Do not inject old responses into a newly reset owner.

## Serial validation

Run **one** command at a time, with no background tests:

```sh
./scripts/test-tensorcore-memory.sh fence
./scripts/test-tensorcore-memory.sh isolation
./scripts/test-tensorcore-memory.sh cmd-e4m3-f32
./scripts/test-tensorcore-memory.sh cmd-nvfp4
./scripts/test-tensorcore-memory.sh cmd-single-mxfp4
```

All nine profiles accept the `cmd-` prefix. Single-port command runs support
E4M3→FP32, MXFP4 and NVFP4. They deliberately use only a two-cycle engine delay
even with sub-bank write queues: the frontend must wait for actual bank idle.
The numeric fixtures are the same author-model-checked data used by PE/mesh
and memory tests, including full raw C, in-place output, fusion boundaries,
dirty FP4 upper nibbles and FP16 slot upper bits, special values and zero-C.

The isolation test additionally queues younger legacy commands, delays older
local writes, holds completion visibility, presents conflicting host reads and
writes, and stalls the CPU response. Host traffic stands for bank-level legacy
access; it is **not** a TileLink DMA, ROB or full RoCC system simulation.

On this host, tests select a single generated C++ translation unit to avoid a
Verilator 5.052 split-build PCH incompatibility with chiseltest 0.6/GCC. This is
test-harness compatibility only, not an RTL or arithmetic change. Test JVMs
remain at 1536 MiB with one active processor; native builds are sequential.

```sh
mill --no-server -j 1 tensorcore.runMain gemmini.tensorcore.GenerateCommandFrontend nvfp4 f32 out/command-frontend
mill --no-server -j 1 integrationSyntax.compile
```

The second command is parse-only. Neither command typechecks or elaborates the
complete Rocket/Chipyard/HardFloat design. Actual results and remaining release
gates are recorded in the parent `docs/progress/`.

For real-source typechecking, top-level RTL and RoCC/MVIN/MVOUT validation,
use `scripts/setup-tensorcore-integration.sh` followed by one
`scripts/test-tensorcore-integration.sh` command at a time. This is distinct
from both parser-only checking and the component tests described above.
