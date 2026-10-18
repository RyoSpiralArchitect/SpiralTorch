# HF Geometry Trainer Qualification

Source candidate on `db12731332c1a5f3fc7d571ab0f33e7f9275e264`, not a package
release or language-quality comparison. [API and boundaries](../../../docs/python/hf-geometry-training.md).

Final installed-wheel qualification: **88 passed, no skips**, including the real
MPS test; three expected MPS pin-memory warnings. Rust artifact tests: **6 passed**.
Native/Python/WASM agree on the pinned 13-placement layout ID. Standard st-nn
clippy passes with 24 existing warnings; this is not a warning-free lint claim.
The installed Python implementations match the reviewed source hashes. See
[validation.json](validation.json) for source, binary and retained-log identities.

## Scope

Real HF Trainer execution with local, randomly initialized tiny GPT-2 and Llama:
four layers, width 8, vocabulary 32, six-position inputs. Four distinct native
geometry families are inserted at explicitly chosen MLP outputs: learnable-radius
WaveGate, Topos, anchored Elliptic, and angle/gain Fractional history. Base weights
are frozen. This is a workflow qualification, not pretrained-model FT evidence.

The CPU continuation test uses four AdamW updates, batch size 2, accumulation 2,
dropout 0.1 and an active clipping threshold of 0.0001. It compares uninterrupted
training with stopping at update 2 and resuming with the same four-update
schedule. The test checks exact geometry, optimizer, scheduler, CPU RNG and
prediction equality, unchanged base weights, and updates in all four geometry
families. A separate isolated Python process also resumes the checkpoint.

The MPS test uses three updates with dropout, stops after two, and requires exact
device RNG, geometry and optimizer equality after resume. It also rejects a
missing MPS RNG companion. CUDA is not exercised in this qualification.

The shared synthetic fixture pins the new Rust-owned layout identity across
native Rust, installed Python and a real wasm32 NN build running in Node. WASM
validation is not browser execution of an HF model.

## Bugs Kept Visible

- An initial command installed an older wheel while its replacement was still
  building. The stale-wheel test log is retained but is not final qualification.
- Direct evaluation exposed HF 4.57.6 counting the unpredicted first label. The
  helper now delegates counting of shifted targets to HF. The test still checks
  agreement with a manually attached model; no mismatch tolerance was added.
- The first geometry-vs-bare evaluation test used identity-initialized adapters;
  it now gives the adapter explicit nonzero state to exercise the insertion.
- A rejection test using `TrainingArguments(torch_compile=True)` leaked
  Accelerate's compilation environment into subsequent tests. Test isolation
  and an actual accelerator-backend guard now cover that separate enable path.
- Independent review found unequal-mask smoothing weights, absent MPS RNG
  persistence, and silently unexecuted insertion sites. Each has a regression
  test and implementation fix, described in [REVIEW.md](REVIEW.md).

Logs and generated HF checkpoints remain private. Public records contain results,
source/binary/log hashes and reproduction commands, not corpora or learned state.

## Reproduce

Build/install this source wheel in an isolated environment with Transformers
4.57.6, Torch >= 2.6, Accelerate and pytest. From the repository root:

```sh
cargo +1.99.0 test --locked -p st-nn --no-default-features --lib geometry_artifact -- --test-threads=1
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
  python -I -B -m pytest --import-mode=importlib --confcutdir=bindings/st-py/tests \
  --rootdir=bindings/st-py/tests -q bindings/st-py/tests/test_hf_geometry_trainer.py \
  bindings/st-py/tests/test_geometry_artifact.py bindings/st-py/tests/test_geometry_adapters.py \
  bindings/st-py/tests/test_geometry_optional_dependency.py
cargo +1.99.0 build --locked -p spiraltorch-wasm --target wasm32-unknown-unknown --release --no-default-features --features nn
wasm-bindgen target/wasm32-unknown-unknown/release/spiraltorch_wasm.wasm --target nodejs --out-dir /tmp/geometry-trainer-wasm
node bindings/st-wasm/tests/geometry_artifact.cjs /tmp/geometry-trainer-wasm/spiraltorch_wasm.js
```

Use wasm-bindgen 0.2.129 for this lockfile. The MPS test requires actual Apple
hardware and is explicitly skipped elsewhere. No network model/data download.
Neither a passing test nor a geometry hash seals the other HF files or verifies
the live base. Complete checkpoint provenance, runtime/data/training identity,
selection and promotion remain executor-owned. No language-quality, throughput,
distributed recovery or whole-checkpoint crash-atomicity claim is made.

## Later-Epoch Resume Follow-Up

The original 88-test qualification above is retained unchanged. A subsequent
offline pretrained GPT-2 probe exposed a missing case: resuming *inside a later
epoch* did not preserve the permutation. Its checkpoint weights and CPU/MPS RNG
matched the uninterrupted run at update 8; divergence began after restoration.
Accelerate 1.14.0 constructs a new `DataLoaderShard` with iteration zero inside
`skip_first_batches`, resetting its nested seedable sampler to epoch zero.

The Trainer now composes Torch's random sampler with a private CPU generator,
seeded by `data_seed` (or `seed`) plus the restored Trainer epoch. It does not
fork the HF loop or patch Accelerate globally. Tests cover epoch boundaries and
resumes at updates 7, 8 and 15, sampler-order parity, and actual batch skipping.
Finite deterministic map data is required; length-grouping, streaming and
epoch-only callback stopping are rejected. Whole-run update-boundary stops are
supported. Guards cover step, logging and evaluation callbacks before writes,
and check again after the parent hook so a no-save stop cannot silently advance.

The follow-up includes 38 Trainer tests against the rebuilt installed candidate,
plus an independent read-only review. Its hashes and qualification scope are in
`resume-followup.json`; that wheel also contains the separately stacked document
dataset feature. The original binary/source hashes are not overwritten.
