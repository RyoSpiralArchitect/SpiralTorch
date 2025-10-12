# Heuristics Precedence

1. **SpiralK** (env `SPIRAL_HEUR_K`) — experimental & fast iteration.
2. **WASM tuner** exported table (`backend/wgpu_heuristics_generated.rs`).
3. **Fallback** — safe defaults (1CE / conservative params).

## SpiralK examples
```text
u2:(c>32768)||(k>128);
wg:sel(sg,256,128);
kl:sel(k>=64,32,sel(k>=16,16,8));
ch:sel(c>32768,8192,sel(c>16384,4096,0))
```
