# Independent Review

A read-only local sub-agent reviewed the Rust window contract, Python/WASM
bindings, public dataset, offline example and tests. It did not edit files,
build, download models, train, mutate Git state or examine the separately
review-blocked PR #2272.

## Findings Closed

1. Python-to-Rust sequence conversion allocated before the document budget.
   The shared Rust limit is now checked before bounded indexed extraction;
   oversized and misleading-iterator regression probes cover the entry.
2. The native plan made the dataset unpickleable under spawn. Serialization now
   reconstructs the plan through Rust; pickle and actual spawned-worker tests
   preserve row boundaries.
3. NumPy/Torch boolean scalars could become token IDs. They are rejected before
   integer conversion, without adding a Torch import dependency to the dataset.
4. WASM f64 ABI coercion accepted null/boolean/string indices. The boundary now
   accepts JsValue and requires an actual number before range validation.
5. The example could resume against new data without a retained identity.
   Resume requires an independently retained expected dataset ID at preflight.
6. Python boolean length/context values differed from JSON semantics. They now
   fail explicitly, including boolean native-window indices.
7. A stateful context integer could change between planning and pickling.
   The dataset freezes one validated integer and reuses it for both operations.
8. from_texts materialized/tokenized over-budget sequences before checking the
   limit. It now checks the shared limit and bounds extraction before either.

Focused reviewer probes confirmed the dataset follow-ups; no other actionable
findings remained in the reviewed native boundary arithmetic or label alignment.
The parent Trainer's later-epoch and callback-stop findings were separately
fixed and re-reviewed; see its resume follow-up record.

This is a bounded independent source/probe review, not an exhaustive audit or a
GitHub approval. The main agent ran the rebuilt-wheel 117-test suite, actual
wasm32 checks and offline GPT-2 full/partial/fresh-process continuation. The
reviewer did not independently repeat those training or build results.
