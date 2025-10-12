
# Changelog

## v1.5.2
- Self‑Rewrite: observation log → soft rule synthesis（閾値・冷却・行数・期限のガード）
- Redis: publish (set+TTL) & simple consensus hooks（バケット sg,lg2c,lg2k 単位）
- SpiralK DSL: `def name(args): expr;`（軽量定義）と `SPIRAL_HEUR_K_LAB` の優先読み
- README: 導入/インストール/クイックスタートを再整理

## v1.5.3
- Wheels/CI: cibuildwheel for macOS universal2 and Linux (manylinux_2_28 / musllinux_1_2). Python 3.11–3.14.
- HIP backend skeleton (feature `hip`), with distributed hooks (stubs by default).
- Distributed 3‑stage TopK outline + CPU fallback merger.
