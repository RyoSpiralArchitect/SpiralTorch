## Summary

- [ ] Purpose: what does this change achieve?
- [ ] Scope: which crates/modules are affected?

## Checklist
- [ ] `cargo +nightly-2026-04-15 fmt --all -- --check`
- [ ] `cargo clippy --workspace --all-targets`
- [ ] `bash scripts/run_rust_clippy_strict.sh`
- [ ] Unit tests pass (`st-core` mandatory)
- [ ] Docs/README updated if behavior changes

## Notes for reviewers
- Perf/Lang implications, tuning knobs, or follow-ups.
