# GPT-2 QKV Held-Out Scoring Preflight

The first read-only comparison of the two-update QKV snapshot against its
frozen LoRA/WaveGate source produced **no quality result**. The 48 previously
scored evaluation documents were not used for the QKV updates, but were not a
fresh blind holdout. The frozen setup used CPU, one Rayon thread, f32 readout,
context length 64, microbatch 1, answer-only labels, and a 10-minute limit.

The first attempt stopped before model loading because the diagnostic script
did not resolve the local Hugging Face config symlink before a regular-file
read. The failure was retained. A versioned second attempt corrected only
that path admission and ran to the 10-minute limit without emitting a paired
receipt. Neither attempt updated weights or produced a usable NLL comparison.

The second attempt's stderr showed first-use CPU matmul autotuning for many
distinct GPT-2 shapes. The existing `SPIRALTORCH_AUTOTUNE=0` path still ran
kernel microbenchmarks and only disabled lookup/persistence. The accompanying
backend change makes that setting bypass selection entirely and use the
existing fixed default kernel. Its unit tests establish control-flow and
matmul correctness; they do **not** establish a measured speedup or language
quality effect. Any later comparison needs a new, predeclared time budget and
must retain these first failures rather than treating a shortened or retried
run as the original experiment.

The original inputs, saved QKV snapshot, protocols and failure logs remain in
the private validation store. This public note contains no model or corpus
payloads. The stored inputs and snapshot are identified in
[the prior qualification](GPT2-QKV-RESUME.md); their SHA-256 hashes identify
retained bytes, not authentication.
