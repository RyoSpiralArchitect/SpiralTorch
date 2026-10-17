# Language Diagnosis Review Corrections

Independent read-only review of PR #2269 at
`2308aaeee17b23a55100546b6ad4b8d8bffbc4d7` found five issues in experiment
orchestration and measurement. This follow-up repairs them without new training,
changing model math, relaxing numerical gates or replacing historical results.

## Repairs

- Existing-process adoption now checks the actual interpreter image and full
  argument list. Runner/output prefixes, changed flags, repository/history paths
  and different virtualenv invocations cannot create a claim. macOS framework
  argv normalization uses the logical launcher marker without resolving away
  the virtualenv path. Only this optional adoption operation requires `psutil`;
  an unavailable inspector fails before claim creation.
- Frozen-weight baselines use the verified completed-update cursor rather than
  future scheduled training batches.
- Both engines must supply the planned number of generation rows and bytes,
  with exact integer bytes in 0..255, before equality can pass.
- Source and report cohort cardinality must agree with all four declared
  cohorts; a shorter source list cannot truncate a `zip` comparison.
- CE is evaluated in shifted coordinates. Uniform float32-representable logits
  near `2^56` or `2^127` retain CE `log(256)` instead of cancellation to zero.

## Revalidation

All **42 offline tests** pass: 17 diagnosis, 15 launch/adoption, six saved-model
qualification and four rechecker source-identity tests. An additional live
OS-inspection check used two owned,
harmless Python processes: the exact process was adopted, while the similar-name
process was rejected before writing a claim. No model was imported by that test.

The original 25-arm budget comparison was recomputed with its original helpers
and private inputs. All 12 input hashes, three checkpoints' complete parameter
and history bindings, seen-only baselines and nine retained generation records
match the published evidence. This does not rerun generation or training.

For the separate nine-model frozen diagnosis, the original producer/helper
identities were checked against retained files and their immutable Git objects,
not the now-newer working tree. Original summaries were verified with the
original checker. The fixed checker then recomputed every summary from the same
raw logits and validated all generation/cohort coverage. All **8,294,400 logits**
pass the unchanged `atol=3e-6`, `rtol=5e-5` gate; maximum derived mean-CE
correction is **8.881784197001252e-16**. Baselines and generation equality are
unchanged. Complete corrected metrics are in
[diagnosis-recheck.json](diagnosis-recheck.json), not substituted for old reports.

The rechecker executes captured checker bytes and rejects checker/rechecker
source drift before publishing, rather than attributing a replay to an
unexecuted revision read from disk afterward.

[review-validation.json](review-validation.json) pins the current source/tests,
retained checks and original publication manifests. The older `validation.json`
files intentionally continue to identify their original Git revision, including
the old scripts and old test counts. Raw corpus, checkpoints, model weights,
logits, original reports and original plans remain unchanged and private where
applicable. No package or model is published or promoted by this correction.

These checks strengthen the evidence, not the learned language behavior. The
reported models still do not beat the simple bigram baseline, and their greedy
output remains space-dominated. No geometry-superiority or speed claim follows.

## Reproduce

```sh
python3 -I -S -B benchmarks/results/2026-10-15-language-learning-diagnosis/test_diagnose.py
python3 -I -S -B benchmarks/results/2026-10-15-language-learning-diagnosis/test_review_recheck.py
python3 -I -S -B benchmarks/results/2026-10-15-language-budget-512/test_run_once.py
python3 -I -S -B benchmarks/results/2026-10-15-language-budget-512/test_qualify.py
python3 -I -S -B benchmarks/results/2026-10-15-language-learning-diagnosis/review_recheck.py \
  --repo "$REPO" --private "$RETAINED_DIAGNOSIS" --output "$NEW_PRIVATE_RESULT"
```

The rechecker requires the original Git commit and retained `plan-reviewed.json`,
`native-reviewed`, `torch-reviewed`, archive and gate. It resolves historical
repository sources from that commit, checks every retained input hash, and
writes exclusively. It never acquires a corpus, runs a model, or rewrites old
evidence. CI and merge status remain separate from these local observations.
