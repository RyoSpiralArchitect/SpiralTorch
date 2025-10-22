#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-all}
MODULE_DIR=${GO_MODULE_PATH:-bindings/go}
LOG_DIR=${GO_LOG_DIR:-logs/go}
mkdir -p "$LOG_DIR"

if ! command -v go >/dev/null 2>&1; then
  echo "Go toolchain not found. Run 'just setup-go' first." | tee "$LOG_DIR/error.log"
  exit 1
fi

if [ ! -f "$MODULE_DIR/go.mod" ]; then
  echo "Go module not found at $MODULE_DIR. Skipping checks." | tee "$LOG_DIR/info.log"
  exit 0
fi

run_lint() {
  local lint_log="$LOG_DIR/lint.log"
  echo "Running gofmt and golangci-lint (if available)..." | tee "$lint_log"
  find "$MODULE_DIR" -name '*.go' -print0 | xargs -0 -r gofmt -l | tee -a "$lint_log"
  if command -v golangci-lint >/dev/null 2>&1; then
    (cd "$MODULE_DIR" && golangci-lint run ./...) | tee -a "$lint_log"
  else
    echo "golangci-lint not installed; skipping." | tee -a "$lint_log"
  fi
}

run_tests() {
  local test_log="$LOG_DIR/test.log"
  echo "Running go test ./..." | tee "$test_log"
  (cd "$MODULE_DIR" && go test ./... -v) | tee -a "$test_log"
}

case "$MODE" in
  lint)
    run_lint
    ;;
  test)
    run_tests
    ;;
  all)
    run_lint
    run_tests
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 2
    ;;
esac
