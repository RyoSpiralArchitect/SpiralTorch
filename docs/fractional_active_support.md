# Exact Active-Support Traversal

Rust now excludes only exact-zero endpoints from GL convolution traversal.
The complete coefficient recurrence, full-K normalization and domain/budget
checks remain unchanged. No tolerance, learned mask, clipped order, shortened
normalizer or new optimizer policy is introduced.

The learning forward visits the union of the coefficient and alpha-derivative
support. At integer orders a zero coefficient can still have a nonzero order
derivative, so pruning from coefficient values alone would be wrong. Input
VJP/JVP may use coefficient support alone. Interior zeros stay in their original
order; accumulation remains f64 with the same checked f32 boundaries. Full
input/control validation runs before any zero-support shortcut.

Dense pre-optimization test oracles compare output and all requested first-order
maps bitwise, including the [2,128,768] LM shape, arbitrary axes, tile boundaries,
signed zeros, subnormals and integer-order tails. General line-operator tests
cover all padding modes and signed scales, not just the learning zero pad.
Python and WASM use the same Rust implementation without reconstructing it.

## Same-Math Measurement

`tools/benchmark_fractional_window.py` compares the buffer-backed Python/Rust
operator with an independent Torch autograd reference. Both normalize over K
before selecting the declared window. Torch uses a window-length conv1d with
the original lag offset, rather than doing needless full-K zero-tap work.
Both scalar pullbacks and optional input gradients are checked before timing;
each route must repeat bitwise during timing. Cross-backend tolerances are fixed
at rtol=atol=3e-5. Parameters-only and input-gradient requests are separate cases.

```sh
python -P -B tools/benchmark_fractional_window.py --window 1 3 --alpha .55 \
  --native-profile release --rounds 12 --output /tmp/window-new.json
python -P -B tools/benchmark_fractional_window.py --window 1 3 --alpha .55 \
  --input-gradient --native-profile release --output /tmp/window-input-new.json
node tools/benchmark_fractional_window_wasm.mjs \
  /path/to/before/spiraltorch_wasm.js /path/to/after/spiraltorch_wasm.js \
  /tmp/wasm-window-new.json
```

Use fresh outputs, a frozen package and the recorded build profile; disable
unrelated Python autopatching. Run before/after builds with the identical script,
inputs, requested gradients and environment. The WASM comparison additionally
checks joint/selective input/parameter VJPs and JVPs across both modules. Node
WASM timings and native/Torch timings are separate scopes. Neither establishes
model speed, geometry quality, browser/GPU residency or library-wide superiority.
Do not rerun or overwrite completed learning studies to create timing evidence.
