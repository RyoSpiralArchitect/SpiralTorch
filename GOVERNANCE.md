# Governance

SpiralTorch is maintained by Ryo ∴ SpiralArchitect with contributions from the
community. Governance is intentionally lightweight so we can iterate quickly
while protecting the project's guiding principles:

1. **Rust-first:** All core features land in Rust before bindings or adapters.
2. **Open geometry:** Z-space constructs must remain auditable and derivative
   works must publish their source (AGPL §13).
3. **Transparent planners:** Kernel plans and heuristics should be inspectable
   and reproducible across backends.

## Decision Making

- Day-to-day decisions are made by the maintainer.
- Substantial changes (new backends, licensing adjustments, governance updates)
  are discussed openly via GitHub Discussions before landing.
- Pull requests require review by the maintainer or a delegate with merge
  rights.

## Becoming a Contributor

Contributions are welcome! Please:

1. Open an issue or discussion describing the change you intend to make.
2. Follow the existing coding style (`cargo fmt`, `cargo clippy`).
3. Ensure all tests relevant to your change pass (`cargo test`, `cargo check`).
4. Agree to license your contribution under AGPL-3.0-or-later.

If you demonstrate sustained, high-quality contributions you may be invited to
help review patches or triage issues.
