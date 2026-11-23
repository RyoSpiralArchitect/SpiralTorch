# Training-only route calibration

This is a finite choice among previously fitted artifacts, not another optimizer
step. See the [frozen study](PROTOCOL.md) and [results](RESULTS.md). Raw decisions,
training-centered decisions and within-frame rankings answer different questions.

Rust users can use the selector independently of the dataset and CLI:

```rust
use st_nn::causal_lm::route_calibration::{CalibrationRole, MatchedRouteCalibration};

let mut selector = MatchedRouteCalibration::<3>::new(CalibrationRole::First)?;
for (origin, candidates) in matched_training_quartets {
    selector.add_frame(&origin, &candidates)?;
}
let report = selector.selection()?;
let chosen_artifact = &candidate_artifacts[report.selected];
```

Each quartet supplies two logits in each of four binary role assignments
(00, 01, 10, 11) on the same frame. Match the origin, mask, teacher and training
order before passing any logits. The selector has no filesystem or model access;
it cannot know whether a caller mislabeled evaluation data as training input.
The winning worst gain can be negative. Nothing is dispatched automatically.

For the experiment, build the lightweight driver from the repository root:

```sh
rustc --edition 2021 -C opt-level=2 tools/native_matched_route_selector.rs -o "$OUT/selector"
ST_ROUTE_SELECTOR="$OUT/selector" python3 -ISB "$STUDY/test_selection.py" -v
```

Use a fresh output directory and the sealed parent sources and journals. Set
`STUDY` to this directory, `BANK` to the parent channel-mix run, `RESIDUAL` to
the earlier residual-metric run, `PREFIX` to the base decoder/LoRA archive, and
`SIGNAL` to the training-signal archive. `design.py OUT BANK RESIDUAL SELECTOR KIND`
seals the inputs. The single-use worker is `run.py OUT BANK PREFIX SIGNAL SELECTOR
KIND INPUT_SHA`; it needs the pinned installed CPU wheel. A newly built selector
has its own binary identity and therefore a different input manifest. Never
invoke the worker twice against the same split.

For a completed run, only replay the journals, with no decoder or new scoring:

```sh
python3 -ISB "$STUDY/audit.py" "$OUT/formal" "$BANK/formal" "$SIGNAL/formal" \
  "$INPUT_SHA" "$NEW/primary.json" "$NEW/orders.tsv"
python3 -ISB "$STUDY/shift_readback.py" "$OUT/formal" "$BANK/formal" "$SIGNAL/formal" \
  "$NEW/primary.json" "$PRIMARY_SHA" "$NEW/shift.json"
```

Output paths must be unused. The second command is a post-hoc readback, not
the preregistered selector or a deployable correction.
