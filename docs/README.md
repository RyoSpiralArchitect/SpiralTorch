# SpiralTorch Docs

- WASM tuner exports `wgpu_heuristics.rs` as a nearest-cluster table.
- Optional `st-kdsl` allows writing a K-like heuristic program at runtime (env `SPIRAL_HEUR_K`).
  Example:
  ```text
  u2:(c>32768)||(k>128); wg:sel(c<4096,128,256); kl:sel(k>=32,32,sel(k>=16,16,8)); ch:sel(c>16384,8192,0)
  ```
- Build with `--features wgpu,kdsl` to enable the K-like override.
