# SpiralTorch v1.7.2 Overlay

**Goal**: *TopK* の **mk（merge_kind）** と **tile_cols** を
- **SpiralK DSL**（実行時ルール）
- **WASM Tuner の生成表**（オフライン実測）
の **二層合意**で決定し、WGPU/HIP/CUDA の全経路から **統一ヒューリスティクス**として利用する。

この overlay は次を含みます：
1. **SpiralK DSL 拡張**：`mk:` と `tile:` のハード指定／`soft(mk,...)` `soft(tile,...)` の柔らかい制約。
2. **SoftLogic（有限領域ソルバ）拡張**：探索領域に `mk∈{0,1,2}`、`tile∈{256,512,1024,2048}` を追加（tile<=cols で制約）。
3. **統一ヒューリ選択（Unison）**：DSL Soft + Redis Soft を合流 → SoftLogic で候補 A を算出。
   さらに **生成表（WASM Tuner）** 候補 C と合成し、二層「合意」ポリシーで最終決定。
4. **生成表サンプル**：`wgpu_heuristics_generated.rs`（Piecewise の簡易モデル）。
   `tools/tuner/gen_generated_rs.py` で JSON（ブラウザ実測）から再生成可能。

---

## 使い方（最短）

1) オーバレイ適用
```bash
unzip -o spiraltorch-overlay-v1_7_2.zip
```

2) SpiralK で mk/tile を制御（任意）
```bash
export SPIRAL_HEUR_SOFT=1
export SPIRAL_HEUR_K='
  # mk: 0=bitonic, 1=shared, 2=warp
  mk: sel(sg && (k<=128), 2, sel(k<=2048, 1, 0));
  tile: sel(log2(c)>15.0, 2048, sel(log2(c)>13.0, 1024, sel(log2(c)>12.0, 512, 256)));
  soft(mk, 2, 0.25, sg && (k<=128));
  soft(mk, 1, 0.20, (k>128)&&(k<=2048));
  soft(tile, 2048, 0.20, log2(c)>15.0);
  soft(tile, 1024, 0.15, (log2(c)>13.0)&&(log2(c)<=15.0));
'
```

3) 生成表（WASM Tuner 出力）での上書き（任意）  
`tools/tuner/tuner_results.json` を置いて生成：
```bash
python3 tools/tuner/gen_generated_rs.py tools/tuner/tuner_results.json   > crates/st-core/src/backend/wgpu_heuristics_generated.rs
```

4) ビルド（例：WGPU）
```bash
cargo build -p st-core --features wgpu,logic,kdsl,kv-redis --release
```

---

## 二層合意（Consensus）ルール（実装方針）
- 候補 A：SoftLogic の最良スコア（DSL/Redis の Soft を加味）
- 候補 B：DSL が **ハード指定**した mk/tile を尊重（不足フィールドは生成表 or SoftLogic で補完）
- 候補 C：WASM Tuner 生成表の推奨（環境に近い実測値の「既定」）

デフォルトは以下：
1. DSL の **ハード指定**があれば最優先（B）。
2. なければ **A vs C** を SoftLogic スコアで比較し、
   - `SPIRAL_HEUR_GEN_WEIGHT`（既定 0.1）を C に加点して **既定を少しだけ贔屓**しつつ勝った方を採用。
3. 採用結果がローカルで有意に勝てば（Wilson CI 下限>0.5 & 最小試行）`Self‑Rewrite` で `heur.kdsl` へ **soft(...) を追記**。

---

## WASM Tuner → 生成表の再生成
- ブラウザ側 / Node 側で実測して、`tuner_results.json` を作る。
- `tools/tuner/gen_generated_rs.py` が piecewise 近似の Rust コードを出力。

`tools/tuner/tuner_results.json` 例:
```json
[
  {"rows": 1024, "cols_min": 4096, "cols_max": 8191, "k_max": 128, "sg": true,  "mk": 2, "tile": 512},
  {"rows": 1024, "cols_min": 8192, "cols_max": 65535,"k_max": 2048,"sg": true,  "mk": 1, "tile": 1024},
  {"rows": 1024, "cols_min": 65536,"cols_max": 262143,"k_max": 4096,"sg": true, "mk": 1, "tile": 2048},
  {"rows": 1024, "cols_min": 4096, "cols_max": 65535,"k_max": 2048,"sg": false, "mk": 1, "tile": 1024},
  {"rows": 1024, "cols_min": 65536,"cols_max": 262143,"k_max": 4096,"sg": false,"mk": 0, "tile": 2048}
]
```

---

## 注意
- mk/tile の実適用は TopK 実装（WGPU/HIP/CUDA）の分岐で使用。WGPU は 1CE/2CE/tile を既実装パスに配線するだけ。
- 生成表が無い場合は、従来通り **SpiralK / SoftLogic / Redis** で決定される（フォールバック安全）。
