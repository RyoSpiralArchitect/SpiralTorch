# Pre-Training Independent Review

A read-only local reviewer inspected preparation, input preflight and the
thin executor before any book LM update. It did not run models, change files,
acquire data or inspect unrelated PR #2272. The declared source was frozen in
`7f8d30e88`; the bounded follow-up below was then independently checked.

## Findings And Dispositions

1. Nested illustration metadata survived the initial regular expression.
   Replaced it with balanced bracket removal and an unterminated-block check.
   Regression coverage preserves non-illustration bracketed prose.
2. An unpinned chapter manifest could replace a selected chapter and its digest.
   Require a separately retained manifest hash before reading any chapter.
3. A symlinked split directory could expose reserve bytes before a hash failure.
   Reject split/leaf symlinks and noncanonical resolved paths before opening.
   A guarded-open regression confirms reserve bytes are not read.
4. Replay established a new package baseline instead of binding the qualified
   runtime. Preflight now records installed Python/native SHA-256 identities;
   replay compares the complete observed inventory against that pinned record,
   then checks it before and after each case. Actual package drift was not alleged.
5. Caller-relative paths changed meaning under the child's repository cwd.
   Resolve every child input/output path before launch; regression assertions
   cover model, geometry, train/evaluation inputs and output.

The final bounded follow-up closed both remaining executor findings with no
additional correctness findings. The reviewer reported 11 targeted stdlib
checks, including package mutation/addition/deletion and path-resolution probes.
Main-process tests independently passed 18 study tests plus the existing four
case-retention tests. These checks do not establish model quality.

## Preflight Evidence

The first all-arm bitwise-gradient assumption failed before model loading or
training. Native radius VJP intentionally uses wider arithmetic. Rather than
lower precision or relax the gate, the revised plan declares two within-path
comparison pairs. All original failures and source hashes remain retained.

Actual tokenizer/synthetic preflight was repeated after input and package-binding
fixes: both pairs have exact initial outputs and parameter gradients; full
tokenized dataset identity is unchanged. The complete private receipt and its
compact public projection have distinct hashes, recorded in
[preflight-qualified.json](preflight-qualified.json).

Separately, parent PR #2277 had a strict Rustdoc failure from bracketed formula
text. The one-line code-span repair `f55042a4892a971e4d426f4a6f817dd125a96e67`
was inspected and passed the same local strict Rustdoc command. No numeric
implementation changed. Latest-head CI remains mandatory for integration.

Review is bounded to this experiment and its narrow parent documentation fix,
not authentication of the environment, a distributed scheduler audit, or review
of every transitive HF/Python/native dependency.
