# Q3: 2×2 vs 4×4 on a fixed global 4×4 task

18 measured performance points; IEEE FMA and both slot-reuse profiles, K=64/128/256.
The reassembled global A/B/C and all expected outputs have the same SHA256 across sizes:
`3a01422314e9ec31fcfb947b4d3e23e981e4c12e58c2264e5cec6bd977b5eeaa`.

Regenerate the comparison from these snapshots, from the submodule root:

```sh
MPLCONFIGDIR=out/dac-q3/scaling/mpl-cache ../.venv/bin/python experiments/dac/scaling.py out/dac-q3/scaling experiments/dac/results/q3-dim2 experiments/dac/results/q3-dim4
```

`scaling.csv` contains direct RTL task timing and reused/new SKY130 areas.
At K=256, the 4×4 core gives 3.62× IEEE / 3.65× Hopper and Blackwell cycle speedup,
while area increases 3.96–3.97×. Throughput per area is about 8–9% lower.
This compares experimental core execution, with no host, SRAM, clock or power model.
8×8/16×16 and WS have not been measured.
