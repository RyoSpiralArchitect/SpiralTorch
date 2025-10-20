# SpiralTorch Safety & Governance

This document outlines the safety governance workflows introduced for the SpiralTorch inference surfaces.

## Governance Overview

The `spiral-safety` crate defines the canonical policy engine for all runtime surfaces. Every inference request is evaluated twice:

1. **Prompt intake** – User supplied content is scored for toxicity and bias using keyword heuristics that flag high-risk content.
2. **Candidate response** – Model outputs (or user-supplied candidates in evaluation scenarios) are re-scored prior to release. Responses that cross the configured threshold are replaced with a refusal message.

Safety verdicts and refusal reasons are stored inside the shared [`AuditSink`](../../crates/spiral-safety/src/lib.rs) which is surfaced through the Python bindings. Audit records capture the channel (prompt or response), the preview of the analysed content, and the policy verdict.

## Workflow

1. **Pre-deployment evaluation** – Run `cargo test -p spiral-safety` to execute the reference dataset contained in `tests/policy_eval.rs`. The suite guarantees that harmful exemplars continue to be blocked while benign prompts are accepted.
2. **Runtime auditing** – Use the Python API `spiral.inference.InferenceClient.audit_events()` to retrieve structured `SafetyEvent` entries for compliance review. Each entry contains the ISO8601 timestamp, channel, preview, and serialized verdict information.
3. **Refusal handling** – The inference runtime automatically injects policy-compliant refusal messages when a prompt or response fails the threshold check. Downstream services can inspect `InferenceResult.accepted` to branch appropriately.

## Audit Logging Hooks

The Rust bindings expose an `AuditLog` handle which mirrors the internal `AuditSink`. The Python helper wraps this handle and provides ergonomic `SafetyEvent` dataclasses for analytics pipelines. Clearing the audit log through `InferenceClient.audit_log.clear()` is gated, ensuring that automated jobs can checkpoint reviews before truncation.

## Continuous Assurance

* `crates/spiral-safety/src/lib.rs` includes unit tests verifying toxicity and bias detection logic, along with regression guards that fail if policy thresholds are increased without updating the evaluation datasets.
* `crates/spiral-safety/tests/policy_eval.rs` acts as a reproducible benchmark that must pass during CI. It enforces blocking of high-risk utterances and acceptance of innocuous prompts.
* Integrators should wire the new `InferenceClient` wrapper into API frontends to centralise policy enforcement and to ensure the audit trail is persisted for quarterly reviews.

For escalation or policy updates reach out to the Safety & Alignment working group.
