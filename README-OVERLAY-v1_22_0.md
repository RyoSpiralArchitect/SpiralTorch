# SpiralTorch Overlay v1.22.0 — SoftRule競合解決 / Beam探索 / 学習層 + フラクショナル微分(テンソル)

この ZIP はレポジトリ直下に **上書き展開** 可能なオーバーレイです（安全のため事前にブランチを切って適用を推奨）。
主な内容：
- **st-logic**: `apply_softmode`（Sum/Normalize/Softmax/Prob）、`beam_select`、（任意）学習ストア `learn.rs`（featureゲート付）
- **st-amg**: Heuristicsへのソフトルールブレンド例（`wgpu_heuristics_amg.rs`）と学習フック（`sr_learn.rs`）
- **st-tensor**: フラクショナル微分（GL 1D, CPU実装 + WGPU骨格）新規 crate（featureゲートでCPUのみデフォルト）

---

## 展開方法

```bash
# レポジトリ直下で
unzip -o spiraltorch-overlay-v1_22_0.zip -d .
```

必要に応じて、`scripts/force_push_overlay.sh` でコミット→タグ→強制プッシュまで一発化できます（内容を確認してください）。

---

## 変更ポイント（要約）

### 1) st-logic（置換/追加）
- `crates/st-logic/src/lib.rs`：
  - `SoftMode`（Sum/Normalize/Softmax/Prob）
  - `apply_softmode(rules, mode)`：SoftRuleの競合解決
  - `beam_select(seed, expand, score_fn, beam_k, max_depth)`：ビーム探索
  - `#[cfg(feature="learn_store")] pub mod learn;` を公開

- `crates/st-logic/src/learn.rs`（新規, feature `learn_store` 下）：
  - A/B結果に基づく **Bandit(Beta)** 更新
  - `.spiraltorch/soft_weights.json` への永続化
  - 依存：`serde`, `serde_json`（※後述の Cargo 追記を参照）

### 2) st-amg（追加/差し替え例）
- `crates/st-amg/src/backend/wgpu_heuristics_amg.rs`（**オーバーレイ例**）
  - SpiralK/生成表の SoftRule を bandit 重みとブレンド、`apply_softmode` で集約
  - `SPIRAL_SOFT_MODE` / `SPIRAL_BEAM_K` / `SPIRAL_SOFT_BANDIT_BLEND` で挙動制御
- `crates/st-amg/src/sr_learn.rs`（新規）
  - Wilson固定化と併用できる **学習ストア更新フック**（feature `learn_store` で有効化）

> 既存の `sr.rs` や `wgpu_heuristics_amg.rs` に手元の差分がある場合は、
> 本オーバーレイの内容を参考に必要箇所のみマージしてください。

### 3) st-tensor（新規 crate）
- CPU: `fractional::fracdiff1d_gl` / `fracdiff1d_gl_vjp`
- WGPU骨格: `backend/wgpu_frac.rs` + `wgpu_shaders/frac_gl_1d.wgsl`（**feature `wgpu_frac` 有効時のみ**）

---

## Cargo 変更（最小）

### A) st-logic に学習層を有効化したい場合（任意）
`crates/st-logic/Cargo.toml` に以下を追記します：
```toml
[features]
default = []
learn_store = ["serde", "serde_json"]

[dependencies]
serde = { version = "1", features = ["derive"], optional = true }
serde_json = { version = "1", optional = true }
```

### B) st-tensor をワークスペースに追加（任意）
ルート `Cargo.toml` にワークスペースメンバとして追加：
```toml
[workspace]
members = [
    # 既存メンバ…
    "crates/st-tensor",
]
```

`crates/st-tensor/Cargo.toml` は本ZIPに含まれています。デフォルトは **CPUのみ** でビルドされます。  
WGPU骨格もビルドする場合は（ワークスペースの wgpu バージョンに合わせて）以下を有効化：
```bash
# 例：
cargo build -p st-tensor --features wgpu_frac
```

> ワークスペースに wgpu のバージョンが固定されている場合、
> `crates/st-tensor/Cargo.toml` の wgpu バージョンを合わせてください。

---

## 使い方（2分スモーク）

```bash
# SoftMode / Beam / 重みブレンド
export SPIRAL_SOFT_MODE=Prob
export SPIRAL_BEAM_K=8
export SPIRAL_SOFT_BANDIT_BLEND=0.35

# 既存 example の起動（st-amg）
cargo run -p st-amg --example pcg_amg
```

### フラクショナル微分（CPU）
```rust
use st_tensor::fractional::{fracdiff1d_gl, fracdiff1d_gl_vjp};

let n = 128usize;
let h = 1.0f32 / n as f32;
let x: Vec<f32> = (0..n).map(|i| i as f32 * h).collect();
let y = fracdiff1d_gl(&x, 0.5, h, n-1);
assert!(y.iter().all(|v| v.is_finite()));
```

---

## 免責
- `wgpu_heuristics_amg.rs` と `sr_learn.rs` はオーバーレイ例です。手元の実装差分がある場合は **必要箇所のみ** マージしてください。
- `st-tensor` の WGPU カーネルは骨格です。まずは CPU 実装と数値一致を最小目標にしてください。

