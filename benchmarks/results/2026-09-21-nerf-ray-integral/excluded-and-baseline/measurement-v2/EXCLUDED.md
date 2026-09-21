# Excluded Startup

No measurement workers ran. The raw f64 JSON-decoding identity check rejected
native/Node representations such as `1.0500000715255735` and
`1.0500000715255737`. Both encode the same declared f32 tensor element.
Parameters and tensor inputs are now compared by exact f32 little-endian hashes,
with exact shape/config checks, rather than by incidental JSON/f64 formatting.
`analyze-before-f32-metadata.py` preserves the failed validator. No numerical
tolerance or condition was removed. PyTorch bounds are also explicitly decoded
as f32 before promoting to f64 for integration, matching the real Rust tensor.
