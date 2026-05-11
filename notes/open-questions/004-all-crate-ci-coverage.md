# All-Crate CI Coverage

## 文脈
While running `/yield` across all SpiralTorch crates, the workspace inventory
showed 34 Cargo workspace members, with 13 members outside `default-members`.
The local full check command `cargo check --workspace --all-targets` reached
non-default packages such as `spiral-selfsup` and stopped because `protoc` was
not installed for the `tboard`/`prost` build path. The existing CI matrix builds
and tests `st-core`, selected upper-stack crates, WASM, wheels, and lint jobs,
but does not appear to have a dedicated all-workspace all-target check job.

## 問い
Should SpiralTorch add a dedicated all-crate CI check, or keep all-crate checks
as a local/manual maintenance surface?

## 選択肢
- A: Add a pull-request CI job for `cargo check --workspace --all-targets` with
  system dependencies such as `protoc` installed. This maximizes crate-wide
  regression visibility, but may make CI slower and surface optional/bindings
  failures on every pull request.
- B: Add a scheduled or manually triggered all-crate job. This keeps routine PR
  latency lower while still giving maintainers periodic full-surface feedback,
  but regressions can sit between scheduled runs.
- C: Keep all-crate checks local/manual and document them as maintainer
  preflight. This avoids CI cost, but makes non-default crates easier to drift.

## 見立て
Lean B. The workspace is large and includes optional bindings/backends, so a
scheduled/manual full check gives useful pressure without turning every small
PR into an environment-heavy gate.
**Important: this is only a record; do not implement from this lean without a decision.**

## 依存
Any CI change that adds, removes, or strengthens all-workspace all-target
coverage depends on this decision.

## 参照
- `Cargo.toml`
- `.github/workflows/ci.yml`
- `docs/development/workspace_crates.md`
- `tools/list_workspace_crates.py`
- `crates/spiral-selfsup/Cargo.toml`
- `CONTRIBUTING.md`

## 関連論点
なし

## 可搬性メモ
Fresh agents should distinguish `default-members` from the full workspace. The
full command currently requires `protoc` because `spiral-selfsup` depends on
`tboard`, which builds protobuf code through `prost` under `--all-targets`.
