# Clippy: 段階 strict 化

SpiralTorch では `cargo clippy -- -D warnings` を一気に workspace 全体へ適用するのではなく、
**“strict 対象クレートを少しずつ増やす”** 方式で導入します。

## 使い方

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

CI は nightly で以下を実行します:

- `cargo clippy --workspace --all-targets`（warn 許容）
- `scripts/run_rust_clippy_strict.sh`（strict subset）
