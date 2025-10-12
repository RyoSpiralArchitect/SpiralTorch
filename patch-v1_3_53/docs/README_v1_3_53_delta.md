
### v1.3.53 Notes (Delta)

- **WGPU where_nd (segments upload)** is now wired: Python enumerates segments → WGPU uploads with
  `queue.write_buffer` at offsets (hole‑skipping), then dispatches the real WGSL where kernel.
- **TopK (WGPU)** remains CPU fallback in this snapshot; autotune scaffold is included.

(Apply this delta to the README Known Issues section accordingly.)
