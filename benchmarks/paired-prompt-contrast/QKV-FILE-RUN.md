# QKV File Training And Inference Handoff

Status: Rust and Python source execution verified on synthetic CPU fixtures.
No pretrained-model study, GPU, cache deletion, push or wheel publication.

The shared checkpoint publisher now accepts either string or byte payloads.
Its atomic-create and transaction behavior is unchanged. QKV optimizer snapshots
and corpus sessions expose `save_checkpoint_new`; sessions also expose
`advance_to_checkpoint_new`, which clones a candidate, executes the bounded
segment, publishes its binary checkpoint and only then commits the live owner.
The 1..=1024 interval rule is shared with existing adapter trainers.

`spiraltorch.native_qkv_runs.run_qkv_corpus` and the new file example orchestrate
initial anchors, absolute checkpoint intervals, explicit resume and final QKV
snapshot export. Rust still owns the entire schedule and arithmetic. The CLI
accepts encoded corpora, a complete explicit recipe, and optional pinned frozen
geometry or LoRA/geometry session sources. Resume rejects a replacement recipe.
`qkv.bin` feeds the unchanged native generation example.

## Verification

Two native tests check binary file/in-memory equality, segment/manual-update
equality, pinned restore, optimizer export, refusal to overwrite existing paths
and dangling links, invalid boundaries and Unix 0600 files. A real late
publication failure is injected through the same shared transaction after the
candidate performs an update; the original state and occupied file remain
unchanged. A numerical candidate failure creates no checkpoint and advances no
live state. Existing adapter checkpoint tests remain in the same suite.

Three Python tests cover the full run entry, not only helper calls. Six cases
(frozen geometry or LoRA/geometry, each with f32/f32-wide-backward/f64 readout)
compare continuous execution, stop/resume and direct API calls. They require
equal reports, session checkpoint bytes and optimizer snapshot bytes. Each
export is passed through the existing cached Z-Space generation entry and must
match direct generation state exactly. Inputs remain unchanged; run files are
0600 and the run directory 0700 on POSIX.

Failure cases include a wrong checkpoint pin rejected before model loading,
changed corpus, conflicting resume flags and existing output preservation. An
injected index-write error after successful native publication stops without
retry, leaves the initial and new checkpoints intact, and emits no final result
or optimizer export. The live session is advanced in this latter case; it is
deliberately not described as multi-file rollback. The run helper retains
partial outputs on errors; these unit tests use disposable fixture directories.

| Check | Result | Wall time |
| --- | --- | --- |
| Offline native decoder tests | 232 passed; no failed/ignored | 18.83 s; bodies 2.13 s |
| CPU binding check | Passed | 15.79 s |
| One offline CPU extension build | Passed | 114.20 s |
| Isolated Python execution | 80 passed; no errors/failures/skips | 2.55 s; bodies 1.114 s |

All invocations passed initially with unchanged source inventories and no guard
triggers. Tests/checks were capped at 180 seconds, build at 600 seconds, Python
at 120 seconds, with a 4 GiB disk floor and two build jobs/no incremental output.
The extension ran in a private 84-file source overlay, not an installed wheel;
the prior saved extension remained byte-identical. There were zero import
attempts or loaded modules for Torch, NumPy, Transformers, Python safetensors
or Accelerate. These checks establish no language-quality or throughput gain.

## Evidence

Private evidence: `2026-11-10-qkv-file-run-v1`. Documentation and the illustrative
recipe were added after runtime checks; the recipe is not a measured FT setting.

| Artifact | SHA-256 |
| --- | --- |
| Tested source inventory | `fcb5a36b9d8c6e5aefb5d3a5f48eae7026b38cddb67f788d1e5c2653079bdafe` |
| Native log | `39b38400b3bbe11caaa4d46ccd805aa615b40a42a2b257fd6de71f8948ec79e6` |
| Binding check log | `0b0c0af76f4f68ae3af9d92909feb54494738c44abbb7a24c4b5ec5c4ae85ea8` |
| Extension build log | `308a6a60657e12b501e3d9a9ecdcac5b1768d825079103412f79f8ee927cfd80` |
| Extension | `f80f9abc5449b80483578ef1d2786f2f8b8cc9d2dd152d727b8631e9685f4ccf` |
| Package manifest | `a0ef310c33a708804636384d9bde0513ca04ec831ff8612d3d670dfaefa7c42a` |
| Python runtime log | `4a852a37a85462d5a3438ca0e83f38ab3158942e3c4498e3c96a62184c384904` |
