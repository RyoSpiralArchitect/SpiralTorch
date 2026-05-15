# Clippy: 段階 strict 化

SpiralTorch では `cargo clippy -- -D warnings` を一気に workspace 全体へ適用するのではなく、
**“strict 対象クレートを少しずつ増やす”** 方式で導入します。

## 使い方

- CI lint job のローカル再現
  - `just ci-lint`
  - `bash scripts/run_ci_lint_local.sh`（`just` なし）
- 全体の現状確認（warn は許容）
  - `just clippy`
- strict（`-D warnings`）で “通す対象” だけチェック
  - `just clippy-strict`

## strict 対象の追加

`configs/clippy_strict_packages.txt` に **package 名** を 1 行 1 つ追加します（空行と `#` コメントは無視されます）。

例:

```text
spiral-config
```

CI の `ubuntu / lint` job は stable channel で以下を実行します（rustfmt だけ pinned nightly）:

- `cargo +nightly-2026-04-15 fmt --all -- --check`
- `cargo clippy --workspace --all-targets`（warn 許容）
- `scripts/run_rust_clippy_strict.sh`（strict subset）

CI の stable channel がローカルより新しいと、ローカルでは見えない Clippy warning が出ることがあります。
その確認には `just ci-lint` を使ってください。
