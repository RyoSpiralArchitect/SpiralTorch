# Exact Detached-Study CI Follow-Up

The original head `7564cb75555e1fec8e9cbdda122b9d6ce4fee7f9` failed its
[macOS WGPU job](https://github.com/RyoSpiralArchitect/SpiralTorch/actions/runs/39096373453/job/117583579863)
in `StudyGpu.test_detached_calls_share_a_recipe_not_mutable_models`.
The first displayed difference was 5.501455783843994 versus 5.501457691192627
in a reported batch loss, with a corresponding bits-per-byte difference. The
large unittest diff was truncated; it does not prove that all other fields,
learned parameters or optimizer state were equal.

This patch **does not claim a numerical root-cause fix**. It preserves exact
serialized equality with no epsilon, skip, retry-until-pass or CI bypass:

- Run four declared two-worker rounds against one sequential reference.
- Report bounded differing JSON paths, including type/shape/encoding changes.
- Only on failure, retain the original synthetic request and all three compared
  reports with SHA-256 identities at an explicitly configured temporary path.
- Upload those synthetic failure diagnostics for one day in CI.

These fixtures contain fabricated tiny byte documents and initial parameters,
not the private HF book corpus, generated text or trained HF artifacts.

## Local Evidence

A separate bounded reproduction made nine calls (one reference plus four
two-worker rounds) through the qualified installed wheel. All eight concurrent
results matched the reference bytes. The updated regression also passed all
four rounds on the local Apple M4 / Metal GPU. Four stdlib diagnostic-helper tests passed,
including the exact observed small numerical mismatch and no-overwrite checks.

Failure was therefore not reproduced locally, not disproven. A passing rerun
alone will not establish a root cause or universal GPU determinism. Latest-head
CI remains required before any merge; retained future failures must be examined.

Retained local evidence IDs:

| Evidence | SHA-256 |
| --- | --- |
| Original CI failed-job log | `58a4ffcff6251720c9c8bb894a9f6babbdabb47cc2997ddc803f24a9de587937` |
| Nine-call reproduction summary | `9ef3f050958a98dd7771946e2e01245cbb493aaffed47889dbe5072bba831e7e` |
| Updated real-GPU regression log | `9a97154cdda1b4377e3ab3ccfce7a8621359776b40a7a47f1ffbe81482db42d8` |
| Four helper-test log | `4c77a5d7573fc8224a97f088e4f5f8f393da911afe1177d73755db78545e68db` |

A read-only independent reviewer found no actionable issues in the helper,
test, CI changes or this note. It confirmed strict equality, exclusive receipt
writes and failure-only bounded retention remain intact. That source review
did not run GPU tests, exercise uploads or independently rehash the evidence.
