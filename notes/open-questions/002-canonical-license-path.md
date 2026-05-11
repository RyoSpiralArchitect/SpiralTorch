# Canonical License Path

## 文脈
The repository's canonical AGPL payload is stored as `LICENSE .txt`, and security tooling explicitly refers to that path. During this pass, the README footer still said `See LICENSE`, which was corrected to link to `LICENSE .txt`. The nonstandard filename may still affect GitHub license detection and newcomer expectations.

## 問い
Should SpiralTorch keep `LICENSE .txt` as the only canonical license file, or introduce a standard `LICENSE` surface as well?

## 選択肢
- A: Keep only `LICENSE .txt`. This preserves existing security-manifest assumptions and avoids duplicate license payload drift, but keeps the repo surface nonstandard for GitHub and new contributors.
- B: Add a standard `LICENSE` file that duplicates or points to `LICENSE .txt`. This improves discoverability, but release/compliance scripts must define which path is canonical and how duplicate drift is prevented.
- C: Migrate the canonical path from `LICENSE .txt` to `LICENSE`. This aligns with ecosystem conventions, but requires coordinated updates across release, security, docs, and compliance hash tooling.

## 見立て
Lean B only if the compliance scripts can enforce a single canonical hash across both surfaces; otherwise A is safer until a release-owner decision exists.
**Important: this is only a record; do not implement from this lean without a decision.**

## 依存
Any change to license file naming, GitHub license-detection cleanup, or release-manifest canonicalization depends on this decision.

## 参照
- `LICENSE .txt`
- `README.md`
- `docs/licensing.md`
- `scripts/security/generate_repo_manifest.py`
- `.github/workflows/repo-license-manifest.yml`
- `.github/workflows/release_wheels.yml`

## 関連論点
なし

## 可搬性メモ
Security tooling currently treats `LICENSE .txt` as intentional. Do not rename or duplicate it casually; first decide how canonical license hashing and release verification should handle the public-facing filename.
