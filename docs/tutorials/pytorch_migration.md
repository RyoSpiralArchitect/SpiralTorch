# PyTorch から SpiralTorch への移行ガイド

SpiralTorch は PyTorch に近い API 感覚で Z-space 固有の最適化やハイパーボリック表現を扱えるよう設計されています。ここでは典型的な PyTorch のコード片に対応する SpiralTorch の書き方を示し、今回追加されたコア NN 機能を中心に整理します。

## 基本的なレイヤー対応

| PyTorch | SpiralTorch |
| --- | --- |
| `nn.Linear(in, out)` | `st_nn::layers::Linear::new("fc", in, out)` |
| `nn.BatchNorm1d(feats)` | `st_nn::layers::BatchNorm1d::new("bn", feats, 0.1, 1e-5)` |
| `nn.BatchNorm1d(feats)` + Z-space | `st_nn::layers::ZSpaceBatchNorm1d::new("bnz", feats, -1.0, 0.1, 1e-5)?` |
| `nn.LayerNorm(feats)` | `st_nn::layers::LayerNorm::new("ln", feats, -1.0, 1e-5)` |
| `nn.LSTM(in, hidden)` | `st_nn::layers::Lstm::new("lstm", in, hidden)` |
| `nn.MaxPool2d` / `nn.AvgPool2d` | `st_nn::layers::conv::MaxPool2d`, `st_nn::layers::conv::AvgPool2d` |

`BatchNorm1d` と `Lstm` は内部で勾配キャッシュとランニング統計を保持しており、`train()`/`eval()` を切り替える PyTorch の流儀を `set_training(true/false)` で再現できます。LSTM の順伝播は行方向に時間軸を持つ 2 次元テンソルを受け取り、隠れ状態とセル状態は `set_state` や `reset_state` で制御します。

## Optimizer とスケジューラ

PyTorch の `torch.optim` に相当する高水準ラッパーとして `ZSpaceOptimizer` と `WarmupCosineScheduler` を追加しました。

```rust
use st_nn::{optim::{ZSpaceOptimizer, OptimizerMode, WarmupCosineScheduler}, layers::Linear, Tensor};

let mut layer = Linear::new("fc", 4, 2)?;
let mut opt = ZSpaceOptimizer::new(1e-2)?;
opt.set_mode(OptimizerMode::hypergrad(-1.0, 5e-3)?);
opt.prepare_module(&mut layer)?; // ハイパーグラッドテープを接続

let mut scheduler = WarmupCosineScheduler::new(1e-2, 1e-4, 100, 1000)?;
let lr = scheduler.step_optimizer(&mut opt, &mut layer)?; // LR を Cosine で更新
```

`OptimizerMode` は Euclidean / Realgrad / Hypergrad を切り替えられ、Z-space 固有の曲率付き学習率を自動的に配線します。スケジューラは PyTorch の `torch.optim.lr_scheduler.CosineAnnealingLR` と同様の挙動にウォームアップを加えたものです。

## 混合精度トレーニング

PyTorch の AMP (`torch.cuda.amp.GradScaler`) に相当する API として `mixed_precision::GradScaler` を追加しました。

```rust
use st_nn::{mixed_precision::GradScaler, optim::{ZSpaceOptimizer, OptimizerMode}, Tensor};

let mut scaler = GradScaler::new(2.0, 2.0, 0.5, 200)?.with_limits(1.0, 1024.0);
let mut opt = ZSpaceOptimizer::new(1e-3)?;
opt.set_mode(OptimizerMode::realgrad(1e-3)?);
opt.prepare_module(&mut model)?;

let scaled_loss = scaler.scale_loss(loss_value);
let stepped = scaler.step(&mut opt, &mut model)?; // オーバーフロー時は自動でステップをスキップ
```

`AutocastGuard` を利用すると演算スコープ単位で自動キャストを無効・有効化できます。

## Grad-CAM との連携

視覚モデル向けには `st_vision::xai::GradCam` が既に用意されています。LSTM や BatchNorm で学習した特徴量に対しても、そのまま Grad-CAM を適用可能です。

```rust
use st_vision::xai::{GradCam, GradCamConfig};
let heatmap = GradCam::attribute(&activations, &gradients, &GradCamConfig::new(h, w))?;
```

## まとめ

* `BatchNorm1d` / `ZSpaceBatchNorm1d` / `Lstm` を含む標準 NN レイヤーが Rust ネイティブで利用可能。
* `ZSpaceOptimizer` + `WarmupCosineScheduler` により学習率制御を SpiralTorch 内で完結。
* `mixed_precision::GradScaler` と `AutocastGuard` が 16bit 近似計算とオーバーフロー検知を提供。
* ドキュメントの対応表を参考にすることで、PyTorch から SpiralTorch への移行がスムーズになります。
