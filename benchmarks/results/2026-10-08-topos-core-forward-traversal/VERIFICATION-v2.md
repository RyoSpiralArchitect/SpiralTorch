# Verification correction (PR #2239)

Use the corrected verification command instead of the historical README's v1
command:

```bash
python -I -S -B benchmarks/results/2026-10-08-topos-core-forward-traversal/verify_v2.py
```

The original `verify.py` checks native shapes, timings and inventory binding,
but omitted each receipt's `parity` assertion. A failed or missing assertion
could pass when its changed receipt was consistently rebound in the inventory.
The original verifier alone is therefore insufficient to check native parity.

The v2 verifier first pins and validates the original `SHA256SUMS` and all four
files it covers, then applies all original checks and requires the exact native
forward/VJP/audit success assertion in every one of the 36 native records. Its
72 negative controls replace or remove that assertion separately in each record
and rebind the inventory; all must be rejected specifically by the parity check.
The first failed and missing controls also reproduce acceptance by v1. The four
original negative controls still run. CI runs this corrected entry point.

This is an offline validation correction, not a rerun or a new performance
measurement. The original measurement bytes, verification record, README,
verifier and checksum manifest are unchanged. The archived success assertions
are checked, not independently re-executed proofs of every native operation.
The WASM screening rejection and decision not to adopt the traversal candidate
remain unchanged.
