# Source Checkout Stub Scope

## 文脈
While running `PYTHONNOUSERSITE=1 python3 tools/run_readme_python_blocks.py --allow-stub-skips` from a fresh source checkout, README examples that use `SpiralSession()` and `plan_topk()` hit native-only planning/session APIs without a compiled wheel. This pass fixed a narrow bug where those calls could degrade into `NameError` by adding explicit stub RuntimeErrors in `spiraltorch/__init__.py`.

## 問い
Should the source-checkout Python stub keep planning/session APIs fail-fast until the native extension is built, or should it implement lightweight CPU/planner placeholders?

## 選択肢
- A: Keep fail-fast stubs for `init_backend`, `describe_device`, `plan`, `plan_topk`, and session construction. This keeps the source checkout honest and avoids inventing planner semantics outside Rust, but README quickstarts remain skip-only without a wheel.
- B: Implement lightweight pure-Python planner/session placeholders. This makes more README examples runnable from a fresh checkout, but it creates a second behavioral surface that must stay aligned with Rust planner semantics.
- C: Split the README examples into source-checkout-safe and wheel-required tracks. This reduces ambiguity for newcomers without adding new stub behavior, but it requires a docs reorganization.

## 見立て
Lean A for code behavior and C for docs. A matches the current native-first architecture, while C would make the onboarding surface less surprising without adding a second planner implementation.
**Important: this is only a record; do not implement from this lean without a decision.**

## 依存
Any follow-up that tries to make `SpiralSession` or planner README examples fully runnable without a built wheel depends on this decision.

## 参照
- `spiraltorch/__init__.py`
- `bindings/st-py/spiraltorch/__init__.py`
- `tools/run_readme_python_blocks.py`
- `README.md` section "Hello SpiralSession quickstart"

## 関連論点
なし

## 可搬性メモ
The repo currently supports a source-checkout pure-Python stub for a subset of APIs, while CI's Python docs smoke builds and installs the wheel before running README blocks. This question is about local source-checkout behavior, not the packaged wheel.
