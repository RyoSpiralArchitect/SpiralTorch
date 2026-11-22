# Cached scoring of complete, terminated answers

The Rust context scores a fixed menu from an already evaluated prompt. A trie
shares equal content prefixes, batches consecutive single-child edges, and
uses every needed vocabulary row, including the row predicting each stop.
Both executors use the same stable full-vocabulary scorer. This differs from
the final-row-only [known suffix API](CACHED_CONTINUATION.md). See the
[Rust, Python and WASM examples](../../../../../bindings/st-wasm/tests/native_causal_lm.md#score-complete-answers).

## One local CPU comparison

Installed release-mode native development wheel, CPython 3.8, macOS 26.4.1
arm64. Deterministic nonconstant WaveGate fixture: 4 layers, width 48, 4 heads,
vocabulary 512, one 32-token prompt, four 6-token candidate contents, one
additional stop each. One menu shares five initial tokens; the other shares
none. One warmup per arm, then all six permutations of the three arms.
Includes Python input and output conversion. Ratios below are medians of
paired ratios, not the ratio of the displayed medians.

| Content | Padded median | Cache incl. prefill | Reused prompt | vs padded incl. / reused | New rows / chunks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shared | 2.355 ms | 0.907 ms | 0.431 ms | 2.57x / 5.47x | 9 / 5 |
| Disjoint | 2.399 ms | 1.412 ms | 0.902 ms | 1.69x / 2.64x | 24 / 4 |

For both menus the maximum absolute difference was zero for token, line,
conditional and accepted-set NLLs, with identical winners. These are
small-model CPU timings, not GPT-2 performance, generation quality, general
latency guarantees, training speedups or GPU results. The trie still needs
one full-vocabulary row per unique prefix, and a causal geometry placement
can replay retained inputs. Sharing reduced decode work but **increased**
live fork storage here: 28,800 versus 14,592 persistent f32 values beyond
the root. Those are logical counts, not measured RSS.

The [raw timings](cache_lines_timings.json) retain every order, duration,
numerical check and work count. Reproduce with the newly built native wheel:
`python -IB cache_lines_bench.py NEW_OUTPUT.json`; no model downloads or
Torch/NumPy. Each run uses new timings rather than expecting identical clocks.

## Independent checks

The Rust LM suite passed 308 tests. It checks the actual head call count,
interior terminal edges, position capacity, root immutability and whole-tree
admission before any new head call. The initial run retained one test failure:
two equivalent singleton reductions differed by one binary64 ULP. Only that
test's scalar-equality assertion changed; the scoring math and other checks did not.
The installed native Python context/decoder suites passed 9 + 70.

A separate older wheel produced the [padded reference](../../../../../bindings/st-wasm/tests/native_cached_lines_fixture.json)
from six existing synthetic models, ordinary and key-lag crossed with no
geometry, trained WaveGate and trained fractional history. Installed Python
matched its scores exactly. Node and headless Chrome each checked 96 token
costs including all stops and 84 invalid requests. Maximum absolute score
differences were 7.12e-8 versus the old padded scorer and 1.78e-15 versus
one full-prefix WASM call per target. No pretrained run or evaluation split
was opened, no weights updated in the scoring comparison, and no release
artifact was published.
