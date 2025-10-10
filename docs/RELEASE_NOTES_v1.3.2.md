# v1.3.2 — Batched GEMM forward (MPSMatrix), 2‑pass ND Reduce, Pool Tuning

- MPSMatrix **batched forward GEMM** (single CommandBuffer loop-encode).
- ND reduce (sum) with **auto 1‑pass/2‑pass** selection.
- Buffer pool LRU with **env** overrides (`SPIRALTORCH_MPS_POOL_MAX_MB`, `SPIRALTORCH_MPS_POOL_MAX_PER_CLASS`).
