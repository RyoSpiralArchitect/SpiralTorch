# v1.3.30

- where_nd: STRIDED (non-contiguous) GPU path + cond u8 packed upload (4/word).
- TopK: adapter-aware k_lane autotuning + pass_k(rem) for final multi-pass stage.
- Python: device="auto" extended to CUDA/MPS probes; capability-aware fallback.
- CI: wheel matrix includes universal2 + musllinux + manylinux_2_28; release job attaches wheels to tag.
