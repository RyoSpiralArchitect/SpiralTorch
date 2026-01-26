# SpiralTorch Strategic Roadmap

_Last updated: 2024-06-XX_

## 1. Major Milestones

| Phase | Target Window | Exit Criteria | Owners | Notes |
| --- | --- | --- | --- | --- |
| Proof of Concept (PoC) Completion | Q3 FY24 | • FFI bindings for top 3 kernels demoed in Python & Rust<br>• Baseline benchmarking harness operational<br>• Stakeholder sign-off on architectural assumptions | Applied Research, Platform | Validate technical feasibility and integration path.
| Performance Validation | Q4 FY24 | • Throughput ≥ +35% vs. current CUDA baseline on reference workloads<br>• Stability tests >72h without regressions<br>• Ops readiness review passed | Performance Eng, QA | Focus on sustained throughput and regression coverage.
| Production Rollout | Q1 FY25 | • CI pipelines updated and cost controls configured<br>• On-call runbooks finalized<br>• Launch approval from Product & Ops councils | Platform, SRE, Product | Gradual canary release with rollback criteria defined.
| Developer Enablement | Q2 FY25 | • SDK docs updated<br>• Internal training sessions delivered<br>• Feedback cycle incorporated into backlog | DevRel, Docs | Ensure teams can safely adopt new APIs.

## 2. Risk Register & Mitigations

| Risk | Likelihood | Impact | Mitigation Strategy | Owner |
| --- | --- | --- | --- | --- |
| FFI stability issues causing runtime crashes | Medium | High | • Establish nightly fuzzing of FFI boundary<br>• Add ABI compatibility tests to CI<br>• Document supported toolchain versions | Platform Eng |
| CI cost increase due to added performance suites | High | Medium | • Tier CI workloads (smoke vs. full performance)<br>• Schedule heavy benchmarks on off-peak runners<br>• Review results weekly to tune coverage | Dev Productivity |
| Long-term maintenance burden on bindings | Medium | Medium | • Adopt shared codegen tooling<br>• Rotate maintainers quarterly<br>• Track dependency updates in release checklist | Bindings Guild |
| Knowledge silos on optimization techniques | Medium | Medium | • Pair programming rotations<br>• Internal tech talks recorded & archived<br>• Publish tuning playbooks in docs/performance | Performance Eng |

## 3. Success Metrics & Tracking

| Metric | Definition | Target | Data Source | Review Cadence |
| --- | --- | --- | --- | --- |
| Performance uplift | % improvement in throughput on canonical inference suite vs. baseline release | ≥ 35% by Performance Validation exit | Benchmark harness stored in `tools/perf-runner` | Bi-weekly during validation phase |
| Developer velocity | Reduction in average integration time for new ops using bindings | ≤ 2 days turn-around by Developer Enablement exit | Jira cycle time dashboard | Monthly |
| Reliability | Mean time between incidents attributed to bindings layer | ≥ 90 days post-production | Incident management tooling | Quarterly |
| CI cost efficiency | $ per successful release candidate build | ≤ 1.2× current baseline after rollout | CI billing reports | Monthly |

**Tracking Approach**

- Centralize metrics dashboards in the internal analytics workspace with automated data pulls.
- Use the roadmap review meeting (every 4 weeks) to present deltas and decide on corrective actions.
- Document metric narratives in `docs/roadmap/status_updates/` following each review.

## 4. Stakeholder Communication Plan

- **Audience:** Product, Engineering Leadership, Operations, Developer Advocacy.
- **Artifacts:** This roadmap document, quarterly status memo, and live demo recordings.
- **Cadence:**
  - Monthly: Roadmap sync (60 min) covering milestone health, risks, and metrics trends.
  - Quarterly: Executive readout summarizing achievements, blockers, and investment asks.
  - Ad hoc: Incident reviews or major decision briefings.
- **Channels:** Slack `#spiraltorch-roadmap`, Confluence summaries, recorded Zoom sessions.
- **Ownership:** Roadmap working group maintains materials, PMO ensures distribution.

## 5. Priority Growth Areas (伸び代が大きい領域)

The roadmap has three high-leverage domains where incremental investment yields outsized returns. These priorities align the technical backbone (WGPU-first performance), backend parity (MPS/CUDA/HIP), and ecosystem readiness (docs, samples, distribution) so each layer reinforces the others.

### 5.1 GPU performance step-up (WGPU-first)

The WGPU stack is positioned for the next performance jump, but it requires deeper shader specialization and runtime adaptation. The biggest levers are:

1. **Subgroup-aware WGSL primitives** for softmax/attention/normalisation hotspots (warp-style reductions on WGPU hardware).
2. **Chimera / Split-K layouts** to keep intermediate tensors in a layout that avoids repeated transpose or layout conversions.
3. **Fusion-first IR** to emit a single kernel for GEMM + pointwise sequences (bias + activation, residual adds).
4. **Runtime telemetry-driven autotuning** to loop kernel selection based on observed execution metrics.

See the [Level 2 GPU optimisation roadmap](../performance/level2_gpu_roadmap.md) for implementation details already framed in the WGPU stack.

### 5.2 Backend parity expansion (MPS / CUDA / HIP)

The [backend capability matrix](../backend_matrix.md) shows multiple placeholders or planned items across MPS/CUDA/HIP. The growth opportunity here is straightforward: each incremental implementation is a net-new feature for users, especially around:

- Tensor ops coverage (including mixed precision paths).
- Autodiff + hypergrad readiness.
- Telemetry and runtime observability.
- Dynamic shapes and graph fusion.
- ONNX export consistency.
- CI coverage and validation pipelines.

Closing these gaps raises reliability and adoption confidence across heterogeneous deployments.

### 5.3 Ecosystem enablement (docs / tutorials / distribution / integrations)

User onboarding and downstream integrations are the fastest multipliers for adoption. Priority investments include:

- Entry-point docs, glossaries, and PyTorch migration recipes.
- Device-switching notebooks, canvas demos, and sample galleries.
- ONNX compatibility, tuning integrations (Optuna, Ray Tune), and early Julia/Go bindings.

These items mirror the [ecosystem roadmap](../ecosystem_roadmap.md) and keep the platform approachable as the core performance stack evolves.

## 6. Next Steps

1. Finalize PoC scope with kernel owners by July 15.
2. Stand up nightly fuzzing pipeline by August 1.
3. Publish baseline performance dashboard by August 15.
4. Schedule first roadmap sync with stakeholders for early September.
