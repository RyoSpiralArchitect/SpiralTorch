
| Feature                          | CPU | WGPU | MPS | CUDA |
|----------------------------------|-----|------|-----|------|
| where_nd (broadcast/strides)     | ✅   | ✅    | ⚠️ stub | ⚠️ stub |
| where_nd (segments upload)       | –   | ⚠️ wiring | ⚠️ wiring | – |
| TopK (k≤/big-K; K-way merge)     | ✅   | ⏳ CPU path | ⏳ | ⏳ |
