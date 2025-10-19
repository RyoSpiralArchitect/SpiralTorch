# SpiralTorchVision ガイド

SpiralTorchVision は SpiralTorch の Z-space ネイティブ機能を拡張しつつ、TorchVision 標準機能との互換性を意識した設計を採用しています。ここでは TorchVision が提供する主要機能と、それを SpiralTorchVision でどのように活用・拡張していくのかを整理します。

## TorchVision 標準機能の整理
- **データセット**: 画像分類、物体検出・セマンティック/インスタンスセグメンテーション、光学フロー、ステレオマッチング、画像ペア、キャプション、動画分類・予測といった代表的タスクを `torchvision.datasets` 経由で提供。
- **モデル**: AlexNet から Vision Transformer 系までの分類モデル、量子化対応分類器、セマンティックセグメンテーション、検出/インスタンスセグメント/キーポイント推定、動画分類、光学フローなど、学習済みウェイトとともに活用可能。
- **Transforms v2**: `torchvision.transforms.v2` は画像・動画・バウンディングボックス・マスク・キーポイントを一貫した API で扱える拡張前処理パイプライン。v1 互換性を維持しつつ高速化。
- **TVTensors**: Image、Video、BoundingBoxes といったテンソルサブクラスによるメタデータ保持と自動ディスパッチを実現。
- **ユーティリティ**: `draw_bounding_boxes`、`draw_segmentation_masks`、`make_grid`、`save_image` などの可視化/保存ツールを `torchvision.utils` で提供。
- **カスタムオペレータ**: `torchvision.ops` による NMS・RoI 系演算、ボックス演算、検出向け損失、Conv/DropBlock/SE など TorchScript 互換のプリミティブ。
- **IO**: JPEG/PNG/WEBP/GIF/AVIF/HEIC のデコードと JPEG/PNG エンコード、動画の読み書き（廃止予定）を備え、高速なテンソル変換が可能。
- **特徴抽出ユーティリティ**: `create_feature_extractor` などで中間特徴を抽出し、可視化や転移学習、FPN などの高度な用途に対応。

## SpiralTorchVision での発展ポイント
- **ZSpaceVolume / VisionProjector**: Z 軸方向の共鳴特徴を蓄積し、Tensor へ崩壊させるボリューム表現。TorchVision モデルの中間表現を取り込んで SpiralTorch の Z-space 解析へ橋渡しする基盤。
- **Differential Resonance 連携**: `st_tensor::DifferentialResonance` と組み合わせることで、TorchVision モデルから得た時空間特徴を SpiralTorch の共鳴フレームへ再投影。
- **将来的な統合**: TorchVision のデータセット/変換を入力段として活かし、Z-space ネイティブな損失や可視化ツール、さらには SpiralTorch 独自のモデルヘッドを追加実装予定。

本ガイドは随時更新し、TorchVision エコシステムと SpiralTorchVision 拡張の対応関係を明確化していきます。
