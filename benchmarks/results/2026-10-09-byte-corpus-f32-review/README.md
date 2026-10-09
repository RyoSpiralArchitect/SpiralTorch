# Float32 corpus-comparison review repairs

An independent read-only review of the final corpus-study integration found two
accepted-input cases outside the canonical output of `prepare()`:

- The geometry delta used the original JSON number instead of its effective
  float32 value. Unchanged `0.3` weights could therefore look like a nonzero
  learned change after upload. Both delta norms now use the float32 baseline.
- The Torch reference inferred parameter dtype. Integer-form values such as
  `[1, 1]` could create integer tensors that cannot require gradients. Reference
  parameters now explicitly use `torch.float32`.

Before repair, a rounding-only comparator control and a real one-update Torch
control with a zero output head both failed to reject unchanged geometry. An
integer-form reference control raised a dtype error. After repair, all 12 tests
pass with CPU PyTorch 2.12.1. The standard-library-only invocation passes 10
tests and explicitly skips the two optional Torch tests. A positive control
also accepts genuine updates from noncanonical decimal initial values.

The original frozen request, Torch reference and native/browser raw outputs
were **not regenerated**. Rechecking each original output with the repaired
comparator passes the unchanged numerical thresholds and reproduces every
value of its published comparison JSON. Canonical prepared initial values were
already float32, so the pilot's results and caveats remain unchanged.

[validation.json](validation.json) records source hashes, control log hashes,
frozen artifact identities and offline comparison results. Raw logs and reports
remain local. This is comparator/reference hardening, not a new GPU run, speed
measurement or evidence of geometry superiority. Earlier records are unchanged.

Run the controls from the repository root:

```sh
python3 -I -S -B tools/test_byte_corpus_study.py -v
python3 -I -B tools/test_byte_corpus_study.py -v
```

The second command requires an installed CPU PyTorch. For offline rechecking,
use the existing `byte_corpus_study.py compare` command with the original frozen
request/reference/report and a new output path. Compare its decoded JSON to the
corresponding original published `native-comparison.json` or
`browser-comparison.json`; no tolerance or stored evidence should be edited.
