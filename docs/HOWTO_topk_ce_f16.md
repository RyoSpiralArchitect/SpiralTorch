
# HOWTO

- TopK unified (WGPU): `WgpuTopKUnified::topk_lastdim(&x_dev, rows, cols, k)`
- f16 CE/NLL (WGPU): `WgpuLseF16` + `WgpuCeF16`
- MPS Pool: use `PoolAutoTune::set_window_secs` / `set_explore_range`
