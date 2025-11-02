# Envelope Tooling Bonus Ideas

このドキュメントでは、ナレーション用エンベロープ処理を支援するためのおまけ開発案をまとめています。いずれも運用者や開発者の負担を下げ、トラブル時の可視化・復旧をサポートすることを目的としています。

## 1. 🧪 WASM-side Envelope Validator GUI
- WebAssembly で動作するバリデーターをブラウザで提供し、エンベロープの静的検証をチェックリスト形式で表示。
- MQ ルートの欠落、ブロックサイズの不整合、必須タグの欠如などを即時フィードバック。
- 運用担当者のセルフチェックツールとして役立つが、気づかれずに裏で支えてくれる存在。

## 2. 📦 Envelope Diff Viewer
- 過去に保存した JSON エンベロープ構成との比較 UI を提供し、タグの増減やルートの変更を差分として提示。
- 運用中の設定変更確認や、デバッグ時の原因究明に役立つナビゲーション機能を備える。
- 夜間のトラブルシュートを支える「妖精」的なアシスタント。

## 3. 📼 Z-space Narration Simulator
- フロントエンドでランダム係数を生成し、WASM を介してナレーション処理を模擬した上で、COBOL 側に送信した“フリ”を行うデモ環境。
- 実運用系を触れない新人教育や、疲れた時のリラクゼーション用途として活用できる。

## 4. 🗃️ Job Envelope Archive
- 発行したナレーションジョブをローカルに保存し、日付やコンテンツで検索可能にするアーカイブシステム。
- 将来的な監査対応やトラブル発生時の復元に備え、安心材料として機能する。

## 5. 👩‍💻 COBOL Function Stub Generator
- エンベロープ定義から COBOL 側のパラメータ定義や CALL 文テンプレートを自動生成。
- キーパーソン不在時の保険として、また COBOL 拡張の実験を後押しするサポートツール。
- `CobolEnvelope::function_stub()` が WORKING-STORAGE/PROCEDURE DIVISION テンプレートを構築し、`CobolDispatchPlanner.toCobolStub()` から WASM 経由でも取得できる。生成結果には MQ/CICS/データセット定義、係数テーブル初期化、`st_cobol_new_resonator`/`st_cobol_describe` 呼び出しが含まれる。 【F:bindings/st-wasm/src/cobol.rs†L741-L1224】【F:bindings/st-wasm/src/cobol.rs†L2283-L2287】【F:bindings/st-wasm/src/cobol_bridge.rs†L500-L503】

## 今後の検討事項
- どの案を優先的に実装するかを、運用部門や新人教育担当とのヒアリングで評価する。
- 各案に必要な技術スタック（WebAssembly、差分ビュー UI、COBOL 自動生成など）の調査とプロトタイピングを進める。
- セキュリティと運用フローへの影響を評価し、導入コストを明確化する。
