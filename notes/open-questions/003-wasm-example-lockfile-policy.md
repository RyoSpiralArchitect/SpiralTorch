# WASM Example Lockfile Policy

## 文脈
While expanding the yield frontier into `bindings/st-wasm/examples/`, the three Vite examples had inconsistent package-lock coverage: `canvas-hypertrain` tracks `package-lock.json`, while `mellin-log-grid` and `cobol-console` do not. This pass only changed `scripts/wasm_demo.sh` to use `npm ci` when a lockfile already exists, and `npm install` otherwise.

## 問い
Should SpiralTorch track npm lockfiles for every WASM example, or intentionally keep some examples lockfile-free?

## 選択肢
- A: Track `package-lock.json` for every WASM example. This improves reproducibility and makes `npm ci` available everywhere, but adds large generated files and requires lock refresh discipline.
- B: Remove per-example lockfiles and rely on semver ranges plus `npm install`. This keeps the repo lighter, but makes example builds less reproducible across time and machines.
- C: Introduce a shared workspace-level package/lock strategy for all WASM examples. This centralizes dependency state, but changes the example layout and requires migration work.

## 見立て
Lean A for demo reproducibility if these examples are meant to be CI/build surfaces; lean C if the number of examples is expected to grow. Do not implement either without deciding the package-management policy.

## 依存
Adding or removing package lockfiles, switching `wasm_demo.sh` to require `npm ci` everywhere, or adding WASM example dependency checks to CI depends on this decision.

## 参照
- `bindings/st-wasm/examples/canvas-hypertrain/package-lock.json`
- `bindings/st-wasm/examples/cobol-console/package.json`
- `bindings/st-wasm/examples/mellin-log-grid/package.json`
- `scripts/wasm_demo.sh`

## 関連論点
なし

## 可搬性メモ
`scripts/wasm_demo.sh` currently installs dependencies inside the selected example directory. After this pass, existing lockfiles are honored with `npm ci`, while examples without lockfiles still use `npm install`.
