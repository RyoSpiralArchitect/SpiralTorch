# Whole-answer scoring without keeping a vocabulary matrix

The prior cached scorer shared decoder work but allocated a full
`unique_prefixes x vocabulary` matrix. The Rust context now reduces each
row immediately to the sorted outgoing logits and the partition of everything
outside those edges. The reverse trie pass still uses the **full** vocabulary.
The original two-part arithmetic, interior stops, conditional costs and tie
rule are shared with the dense scorer. Dense probabilities are still available
where training needs them; the inference-only compact representation is not
used as a substitute for a vocabulary gradient.

## Memory and latency on one machine

macOS 26.4.1 arm64, release-mode CPU wheels, CPython 3.8. The 50,257-token
vocabulary is GPT-2-sized, **not** a pretrained GPT-2 checkpoint. The model
was generated locally: width 8, one layer, two heads and a WaveGate at the
attention output. One eight-token prompt, four disjoint 32-token answers and
one scored stop per answer. Neither data preparation nor prefill is timed.

| Cached implementation | Whole process peak | Increase in high-water mark during scoring | Scoring time |
| --- | ---: | ---: | ---: |
| Previous wheel (`14d30f0`), dense rows | 113.97 MiB | 62.47 MiB | 32.90 ms |
| Streaming, default tile | 54.20 MiB | 2.65 MiB | 30.94 ms |
| Streaming, one-row tile | 52.22 MiB | 0.68 MiB | 29.33 ms |

Medians over six fresh processes per arm, one for each order of the three
arms. The whole-process peak was lower with the default tiles in all six
orders. The middle column is the difference in process-lifetime RSS high-water
marks after/before the scoring call, not the allocator size or instantaneous
RSS. Host math dispatch and caching can affect timings; these are not general
latency or throughput guarantees. The full reference ran in its own nineteenth
process. All 18 measured menus matched that old **padded** scorer exactly,
including the winner, all 132 token costs, complete and conditional NLLs.

All arms decoded 128 content tokens in four decoder chunks. Streaming scanned
6,483,153 vocabulary values (including the precomputed prompt row) but only
132 raw edge logits survived reduction. The largest new projected tensor
contained 251,285 values / 28 head calls by default versus 50,257 values /
128 head calls with a one-row budget. These counters exclude the prompt's
retained logits, KV, per-prefix partitions, activations and kernel temporaries.
The same cumulative logit limit applies **before** any tile is evaluated.

## Reproduction and controls

[All paired results](stream_line_measurements.json) include the order, score
hashes, per-process high-water marks and timings. With two installed native
wheels (previous dense-cache wheel and current streaming wheel), from this
directory, use fresh paths:

```sh
"$OLD_PY" -IB stream_line_bench.py prepare "$FIXTURE"
python3 -ISB stream_line_bench.py compare "$OLD_PY" "$NEW_PY" "$FIXTURE" "$OUT"
```

Rust: 311 LM tests including huge shared logits, underflow, exact ties,
streamed partitions, cumulative-work rejection and independently counted KV
decodes vs head calls. Installed Python: 10 context + 70 decoder tests.
Python reproduced the old six-model artificial padded oracle exactly in
default and one-row modes (96 token costs each). Node and headless Chrome
reproduced both modes with maximum score error 7.12e-8 against that oracle;
the same-engine per-target full-prefix check was within 1.78e-15.

The four geometry-bearing synthetic fixtures additionally ran the dense
set, mixed-menu and accepted-subset forward and gradient APIs on both wheels.
Their 12 serialized records, including all 84 nonzero geometry-gradient
components, matched byte-for-byte. To reproduce, run
`native_cached_lines_fixture.py gradients NEW_OUTPUT` on each wheel and compare.
No real-model parameters were updated or evaluated, and no GPU was used.
See the [public API examples](../../../../../bindings/st-wasm/tests/native_causal_lm.md#score-complete-answers).
