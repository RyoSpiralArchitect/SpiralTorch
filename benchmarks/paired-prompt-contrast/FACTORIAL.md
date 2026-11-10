# Separating presentation from request wording

Use the **same 32 observed development records**, not a new holdout. Values,
field order, target field, reference answer and final `Answer:` line are
identical across the four prompts for each record/field.

| Fields line | `What is the <field>?` | `Return the <field>.` |
| --- | --- | --- |
| `field: value \| field: value` | `pipe_what` (saved) | `pipe_return` (crossed) |
| `Record: field = value; field = value.` | `record_what` (crossed) | `record_return` (saved) |

"Presentation" includes the prefix, separators, assignment notation and final
punctuation of the fields line. "Request" includes its wording and punctuation.
This design does not isolate an individual character. The four strings are
formed by crossing the **original lines**, not regenerating records.

```sh
python3 -I -S -B benchmarks/paired-prompt-contrast/test_factorial.py
python3 -I -S -B benchmarks/paired-prompt-contrast/test_factorial_archive.py
python3 -I -B benchmarks/paired-prompt-contrast/factorial.py --tokenizer /path/to/tokenizer.json
```

The last command admits all 384 prompt prefixes **without a model call**.
All four cells must preserve their prefix under reference/LF tokenization, use
identical distinct first-answer and LF IDs, and have unique full prompt IDs.
The two diagonals reconstruct the prior token matrix exactly. The four-cell
matrix SHA-256 is `7f37c14a9a9816039d9c0ff9ad1456276a647b482f0c26007cadc661ad1826e9`.

`factorial_archive.load_cached` requires an independent completion-file pin,
the base model ID, the checkpoint archive digest, the ordered checkpoint
descriptors and the wheel/native digests. It verifies the parent journal, matches each start/return to the model,
field, form and input hash and reconciles the saved diagonal counts and mean.
`measure_crossed` checks all of one model's cached cells before invoking its
callback. It invokes the callback only for `pipe_return` and `record_what`,
alternating which comes first. No automatic run, model download or release.

## Readout

Let `Y` be the raw first-answer logit minus LF logit, with the full-vocabulary
greedy winner recorded separately. For each field compute:

- **Presentation:** mean of `record - pipe` at each of the two requests.
- **Request:** mean of `return - what` at each of the two presentations.
- **Interaction:** `(record_return - pipe_return) - (record_what - pipe_what)`.

Average the three matched field differences within each record, then give each
record equal weight. Keep the four simple differences and all target/LF/other
winner transitions as well. The two main differences sum to the original
`record_return - pipe_what` diagonal. They need not be additive predictions for
individual cells. Do not select a model by these records or report 384 independent
items, complete-answer accuracy, replication across corpora, or a training gain.

The cached and new cells can come from **different executions**. A common
additive margin offset on both new cells cancels out of both balanced main
differences, but changes the interaction by minus twice that offset. Input- or
model-dependent drift need not cancel. Same weight/binary hashes are not a
same-execution replicate: interpret the interaction conditionally.

## Same-execution alternative

`measure_all` accepts no historical score inputs. It checks the ordered input
grid against its token hash before the first callback, then measures all four
cells together for each record/field. A four-order Williams design gives each
cell each position 24 times and each ordered pair of different consecutive
cells 24 times within the 96 quartets per model. All models use the same order.
Pass `mode="all_measured"` to the readout to reject any evidence marked cached.
For 13 models this is **4,992 new prompt-only reads**, not 2,496. No change to
the trained weights, comparison population or first-token estimand is implied.
