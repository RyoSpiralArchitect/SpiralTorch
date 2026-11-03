# Native observed checkpoint recovery

An offline process-boundary regression test, **not** the interrupted
[book observation study](../2026-10-31-native-geometry-step-observation/README.md).
No book data, pretrained weights, external rescoring or GPU were used.

On macOS arm64/CPython 3.12, the [probe](../../../bindings/st-py/tests/probe_observed_checkpoint_recovery.py)
compared a normal four-update run on the public 0.4.33 wheel with two processes
on a private development wheel. The first candidate process exited deliberately
at update 2 without producing a terminal summary. The next process restored
the receipt-pinned optimizer and model and completed updates 3 and 4.
All four step reports and all four checkpoint payloads were byte-identical
to the public-wheel reference: [hashes](report.json).

| Identity | Value |
| --- | --- |
| Tested source | `333d786b68bb97cc6658abd1758d74f4ecffdfca` |
| Published 0.4.33 macOS universal2 wheel SHA-256 | `1a6ba0054644ce380b0206cc1b8c62932a80e2b4cb989947a4459b7fd399357c` |
| Development arm64 wheel SHA-256 (not a release) | `95a6a6412c54b2beeb6b59806e3532064285829f327efbeac5b79969e6e5e3ee` |
| Build | Rust 1.99.1, maturin 1.15.0, default features plus logic and kdsl |

[Run the probe locally](../../../docs/python/native-corpus-training.md#keep-an-observed-recovery-point-source-after-0433)
with the two installed interpreters and a **new** private output directory.
Separate injected-storage tests reject a checkpoint without its observation
receipt; a previous completed pair remains usable. This checks one tiny native
CPU trajectory, not power-loss durability, cross-device agreement, general
version compatibility, or better language-model quality.
