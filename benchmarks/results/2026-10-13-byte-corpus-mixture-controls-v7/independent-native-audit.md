# Independent Native Artifact Audit

A separate read-only reviewer recomputed the retained native/Torch evidence
with CPU standard-library tools. No files, model outputs or measurements were
modified. No UI, build, network, Torch or GPU work was performed by the reviewer.
This is an artifact-consistency audit, not independent execution attestation.

Verdict: no actionable findings. Numerical learning/parity checks pass, but
learned-versus-fixed BPB differences are tiny and do not support a quality gain.

- All 27 v5 cases preserve their 304,608 initial float32 parameter bits. Their
  full native case reports exactly equal the retained v5 reports, including
  histories and final parameters. Documents, windows, rate and schedule match.
- All 6,528 training losses, 4,896 held-out batch losses and 575,676 final
  parameter scalars satisfy the frozen tolerance. Maximum loss discrepancy is
  `1.430511474609375e-6`; maximum parameter discrepancy is `8.344650268554688e-7`.
- All 54 individual learned-router components pass. Minimum reference update
  magnitude is approximately `5.917e-6`; maximum relative error is
  `7.3476752370586e-5`, below `0.002`. All 54 fixed-router components keep their
  initial bits in both engines. All 192 learned geometry-family checks pass.
- All 12 primary BPB contrasts were recomputed from held-out batch losses,
  including secondary contrasts and discrepancy/sign fields. Maximum absolute
  primary difference is `2.472250608676063e-7` BPB. Six sign classifications
  disagree: three opposite-sign pairs and three pairs where Torch gives zero.
  Maximum native/Torch contrast discrepancy is approximately `7.524e-8` BPB.
- Source/request hashes and canonical component scales match. Recomputed
  reported mixture RMS errors peak at `1.514142244918304e-7`, below `1e-5`.

In-memory counterexamples reject an individual-router violation hidden by
aggregate L2, a frozen `+0` to `-0` mutation, and a missing training revision.

Inputs were `$RAW/input/request.json`, `$RAW/input/verification.json`,
`$RAW/native.json`, `$RAW/torch.json`, `$RAW/native-comparison.json`, and the
retained v5 request/native report. Hashes are listed in the evidence manifest.
First-update controls, resume and browser data were outside this audit. Three
seeds remain three quality samples, not twelve independent samples. The audit
does not independently remeasure model outputs or preparation RMS.
