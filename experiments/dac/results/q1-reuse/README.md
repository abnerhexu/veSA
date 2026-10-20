# Q2: reuse operand slots after replay reads them

Measured on 2026-10-19 on `experiments/dac-q1`. Source and input hashes are in
the numbered manifests; prior data comes from `../q1-overlap/` and
`../q1-initial/`, identified by `baseline-reuse.json`.

The new B2 candidate has one physical raw A/B bank and two sets of logical
block metadata. The next block writes only slots already read by current replay,
including the same slot on the same edge using the old register value. It waits
before overwriting unread operands. C dependence, alignment/truncation, special
flags, block-end rounding, and replay arithmetic are unchanged from dual-bank B2.

| Profile | Dual-bank area (µm²) | Slot-reuse area (µm²) | K=64 cycles | K=128 cycles | K=256 cycles |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hopper | 220,896.86 | 140,432.19 | 105 → 107 | 175 → 177 | 315 → 317 |
| Blackwell | 233,433.88 | 152,601.36 | 105 → 107 | 175 → 177 | 315 → 317 |

Compared with dual-bank B2, area falls 36.43% / 34.63%, with 0.635% more cycles
at K=256. Operations/cycle/area improves by 56.31% / 52.00%. Compared with serial
B1, area is only 5.98% / 2.05% higher, while K=256 takes 317 rather than 560
cycles (43.39% fewer cycles). These are core area/cycle results without an
assumed physical clock.

Raw operands fall from 1024 to 512 bits per PE; the four-PE array saves 2048
register bits. Synthesis sequential area drops by 61,498.9824 µm² in either
profile; the remaining saving comes from combinational logic.

Two selected ScalaTest cases check the same 23 fixed matrices plus a bubbled /
backpressured task and a restart after reset during overlap, per architecture:
50 complete matrices / 200 output words, plus two aborted tasks. The second
block's first input is accepted at cycle 34. No vector regeneration, unchanged
design regression, or CPU million-case sweep was run.

Raw logs, netlists and plots are in `../../../../out/dac-q1/reuse/`.
`summary.csv` includes all seven historical/current variants. The figure focuses
on B0 and the two B2 choices. Power, SRAM/RoCC, node scaling, and other formats
remain outside this measurement. The register reuse depends on old-value
same-edge reads and is not a validated SRAM implementation.
