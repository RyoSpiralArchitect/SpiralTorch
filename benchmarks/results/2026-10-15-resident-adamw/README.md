# Resident AdamW Validation, 2026-10-15

This is synthetic optimizer and resume validation, not a language-quality or
performance benchmark. The implementation and usage contract are in
[resident_adamw.md](../../../docs/resident_adamw.md).

## Results

| Path | Passed | Expected skips |
| --- | ---: | ---: |
| Shared Rust numerical contracts | 126 | 0 |
| Native GPU parameter owner | 7 | 0 |
| Native byte decoder/study/serving | 111 | 0 |
| CPU-only checkpoint contracts | 15 | 0 |
| Installed WGPU wheel: language suite | 32 | 2 CPU-only cases |
| Installed CPU-only wheel: language suite | 20 | 14 GPU cases |
| Installed release smoke, each wheel | 9 | 0 |

These suites overlap; counts are not a number of distinct mathematical cases.
Native execution used Apple M4 / Metal. The actual browser receipt is
[browser.json](browser.json), not a WASM-build-only result.

Both native and browser runs compare 12 updates, weights and both moments against
a CPU PyTorch 2.12.1 fixture. Zero-rate freezing uses PyTorch `grad=None`, not
`lr=0` with a present gradient. They also test same-device bitwise resume,
retained snapshots, whole-owner rejection, exhausted clocks and extreme-number
division guards. NN/Python tests exercise real CE/backward through contextual
geometry, freezing and resumed updates.

A WGPU-trained query-mixture checkpoint with 25 optimizer slots was also read by
the isolated CPU-only wheel. Its exact JSON and optimizer configuration survived;
requesting WGPU execution still raised explicitly. That checkpoint is private.
Its hash and the two installed native-extension hashes are recorded in
[verification.json](verification.json).

## Reproduce

Run the Rust, Python and browser commands in the API document. For the broader
Python suite, use each separately installed wheel's interpreter:

```bash
# WGPU wheel, real hardware enabled.
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 python -I -B -m unittest discover \
  -s bindings/st-py/tests -p 'test_nn_byte_*.py' -v
python -I -B tests/test_release_learning_smoke.py -v

# CPU-only wheel, GPU opt-in unset.
python -I -B -m unittest discover \
  -s bindings/st-py/tests -p 'test_nn_byte_*.py' -v
python -I -B tests/test_release_learning_smoke.py -v
```

The GPU wheel used default features. The CPU-only build used
`--no-default-features --features python-default`; both were built with
`maturin build --locked --interpreter python3` into separate private directories
and installed with `--no-deps`. They retain 0.4.31 metadata, are not release
artifacts, and must not be uploaded over the immutable PyPI release.

Strict backend Clippy passed natively and for WASM with installed Rust 1.99.
The older 1.97 run rejected seven existing uses of one lint name; no warning suppression
or unrelated source change was used to make the newer check pass. The pinned
nightly formatter checked the changed Rust files, not deleted/unrelated files.

Full logs, built wheels, generated web assets and the transport checkpoint remain
local. Their hashes and the checked source hashes are public in the manifest.
Hashes identify retained bytes; they are not signatures or execution attestation.
Earlier development failures are retained locally, not counted as passing runs.
