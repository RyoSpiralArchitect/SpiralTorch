# Nearby Fisher Geometry: Candidate Follow-Up

This repair carries the Rust-owned Fisher-Rao metric into the 0.4.30 Python
candidate and the standalone WASM byte-decoder client. It does not publish
the candidate or replace earlier results with newer evidence.

## Numerical Change

Separately normalized square-root distributions can round away a small logit
change before their distance or derivative is computed. The close-pair chart
instead keeps original logit bits and contracts pair-relative deviations.
Its coordinate-centered covariance separates linear and curvature terms.
Unnormalized key masses also retain their deviations until contraction;
otherwise swapping the query and key loses a second-order gradient.

Regression cases cover perturbations from 2^-30 through the minimum f32
subnormal, common offsets, both endpoint orders, a shared pivot, mixed
close/far pairs and the chart transition. For epsilon=2^-149 and
gain=seed=2^127, let S=gain^2*epsilon^2=2^-44. With a uniform key and query
[0,epsilon,-epsilon], the zero-coordinate key/query gradients are
[-S/9,S/3]; swapping endpoints must produce [S/3,-S/9], not reverse their
signs. Both CPU and actual GPU reproductions of the old failure are retained.

This is a finite numerical repair, not arbitrary-precision exponential
evaluation or an exactness guarantee for every geometric reduction. The
previous [exact router accumulation](../2026-10-13-exact-router-reduction-v1/README.md)
has a different, explicitly bounded stored-operand contract.

## Evidence And Cost

The pinned [validation record](validation.json) separates kernel tests,
installed-wheel tests, independent static review and actual learning runs.
Full scalar outcomes are retained for [native](native-results.json),
[browser](browser-results.json) and [Python](python-results.json). Each uses
the unchanged six-case v2 fixture and 0.002 relative gradient gate, with
16 updates per case. Python also verifies 54 resumed updates exactly,
including derivatives, parameters and mixture weights, not just loss.
The installed package's native extension is checked against the wheel hash.

118 CPU kernel tests and 337 native WGPU tests pass. One pre-existing
explicit timestamp experiment is ignored. Python passes eight API tests;
the CPU-only artifact rejection test is skipped for this WGPU wheel.
Native/WASM Clippy, strict kernel rustdoc and workspace formatting pass.
Independent read-only review found the swapped-endpoint defect; follow-up
source review found no new actionable issue after the repair. Runtime tests
were performed separately, not by the static reviewer.

The close-pair covariance currently costs O(projected_columns^2) per pair.
Its private chart uses 20 bytes per coordinate, the pair cache remains
64 bytes per pair, and the backward buffer uses 9N+H+4P 32-bit words.
One sequential native harness run increased from 19.45 to 66.29 seconds.
These timings include setup and compilation; they are not isolated kernel
measurements or a controlled PyTorch comparison. This is a correctness-first
path, not a performance improvement. Optimizing its repeated contractions
without losing the regression signals remains follow-up work.

## Reproduce

Use the source commits and hashes in validation.json, Rust 1.99.0 for
kernel tests/Clippy and Rust 1.97.0 for optimized native/WASM/wheel builds.
Do not overwrite an older raw-result directory.

```sh
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.99.0 test --locked \
  -p st-kernel-contracts -p st-backend-wgpu --lib -- --test-threads=1
cargo +1.99.0 clippy --locked -p st-kernel-contracts -p st-backend-wgpu \
  --all-targets -- -D warnings
cargo +1.99.0 clippy --locked -p st-backend-wgpu \
  --target wasm32-unknown-unknown --lib -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo +1.99.0 doc --locked -p st-kernel-contracts --no-deps
```

Follow the [frozen v2 native/browser protocol](../2026-10-13-matched-geometry-learning-v2/README.md)
without regenerating a different fixture. Build the Python wheel at the client
source commit using maturin with --release --locked --features logic,kdsl.
Run the validator against the fresh native/browser directory, with WHEEL
pointing to the installed candidate. The isolated interpreter needs only the
tools directory added for the validator's sibling modules, not bindings/st-py:

```sh
python -I -B -c 'import sys,runpy;sys.path.insert(0,"tools");sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name="__main__")' \
  tools/validate_python_byte_decoder.py "$RAW" "$RAW/python.json" --wheel "$WHEEL"
```

Raw tensors, logs, binaries and prior failed attempts stay local; hashes and
all scalar conditions are public. This is macOS arm64 WGPU qualification,
not fresh Linux/Windows/universal2 wheel coverage or proof of LLM quality.
Wave time/batch cancellation and official release gates remain separate
open work. No merge, tag or PyPI publication is asserted here.
