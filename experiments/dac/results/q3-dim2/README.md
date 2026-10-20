# Q3 2×2: fixed global 4×4 task

Measured experimental E4M3/FP32 core; IEEE FMA and Hopper/Blackwell slot reuse.
Nine identical global tasks at both sizes (seed 2899); K=64/128/256 are performance points.
See ../../README.md for commands and cycle scope.

`runtime.csv` contains tiles and separately named bubble/reset cases.
`task-runtime.csv` directly measures first request acceptance to final result validity,
including two interface cycles between output tiles. SRAM/DMA/host costs are excluded.
Raw logs, RTL, mapped netlists and plots remain in `out/dac-q3/dim2/`.
Native SKY130 HD synthesis area; no frequency assumption, power estimate or 16 nm projection.

Only areas and coarse breakdown were reused from Q1; see baseline-reuse.json. New vectors and cycles were measured.
