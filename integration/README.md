# Real Gemmini integration build

This target compiles **all** `src/main/scala` Gemmini sources against real,
locked Rocket/CDE/HardFloat/FireSim targetutils sources. It does not use API
stubs or the parser-only `integrationSyntax` target. The existing standalone
`tensorcore` target remains independent of these downloads.

The source graph is taken from [Chipyard 1.11.0's build](https://github.com/ucb-bar/chipyard/blob/ac58f38d77c99e9d1cafa64dfd6d4b00bdcd43e1/build.sbt),
which uses Scala 2.13.10 and Chisel 3.6.0. `dependencies.lock` pins the four
upstream commits. Only required Scala/resource directories are checked out;
no recursive submodules, CPU software toolchain, Linux image or complete
Chipyard tree is installed. Source caches and generated artifacts stay in
ignored `out/`. Setup refuses to replace dirty or unexpected checkouts, and
the build verifies commit IDs and source cleanliness.

```sh
./scripts/setup-tensorcore-integration.sh
./scripts/test-tensorcore-integration.sh compile
./scripts/test-tensorcore-integration.sh rtl-e4m3
./scripts/test-tensorcore-integration.sh e4m3-f32
```

Run each command to completion before the next; never run tests in parallel.
The runner sets a 1536 MiB heap, one active processor and Mill `-j 1`, including
dependency compiles. The Maven versions and Chisel plugin are explicit in
`build.mill`. Subsequent builds reuse the dependency/compiler cache.

`GemminiIntegrationHarness` contains the actual Gemmini `LazyRoCC`, command
queue, unrollers, reservation station, legacy controllers, TLB, DMA, scalers,
Tensor frontend/engine and production SRAM. Its diplomatic TileLink port
connects to Rocket's TLRAM via TLXbar/Fragmenter. A second real TileLink master
lets tests initialize and inspect system memory without bypassing Gemmini's
local memory. The injected RoCC commands carry machine-mode status. Virtual
memory is disabled and an assertion forbids page walks; VM/PTW correctness
and CPU instruction execution are not claimed by this harness.

Tests load the original author-checked fixtures from `modeling/tensorcore`,
transport raw operands/C/scales with legacy MVIN, then execute Tensor RUN and
read the result through legacy full-width MVOUT. They deliberately avoid an
idle wait between DMA loads and RUN, queue MVOUT behind RUN, stall the CPU
completion response and check system-memory sentinels. A direct bank write,
simplified DMA replacement or CPU numerical computation is not used to run MMA.
The E4M3 regression also exercises counter-response arbitration, a rejected Run
without a D-memory side effect, actual zero-fill, older legacy OS execution and
an in-flight joint reset followed by descriptor reconfiguration/restart.

All nine profile names are accepted by the serial runner: `e4m3-f32`,
`e4m3-f16`, `e5m2-f32`, `e5m2-f16`, `fp16-f32`, `fp16-f16`, `bf16-f32`,
`mxfp4`, `nvfp4`. RTL generation also accepts `rtl-fp16`, `rtl-nvfp4` and
`rtl-disabled`, the last checking that the default-off legacy path elaborates.
Each test uses a 2x2 array, internal SRAM and a 64-bit TileLink beat; no FP32
A/B MMA is added. FP16 C/D still occupies low halfwords in 32-bit raw slots.

All nine profiles and enabled/default-off RTL generation passed P5; see the
parent `docs/progress/2026-10-19-07-integration-build.md` for that evidence.
The [Blackwell v1 release specification](BLACKWELL_RELEASE.md) fixes the exact
supported scope, reference/fixture hashes and P6 serial verification commands.
A full CPU or Chipyard SoC boot is distinct from accelerator-level RoCC/TileLink
validation. No individual profile result implies all release gates have passed.
