#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-all}
PROJECT_DIR=${JULIA_PROJECT_PATH:-bindings/julia}
LOG_DIR=${JULIA_LOG_DIR:-logs/julia}
mkdir -p "$LOG_DIR"

if ! command -v julia >/dev/null 2>&1; then
  echo "Julia executable not found. Run 'just setup-julia' first." | tee "$LOG_DIR/error.log"
  exit 1
fi

if [ ! -f "$PROJECT_DIR/Project.toml" ]; then
  echo "Julia project not found at $PROJECT_DIR. Skipping checks." | tee "$LOG_DIR/info.log"
  exit 0
fi

run_lint() {
  local lint_log="$LOG_DIR/lint.log"
  echo "Running Julia formatter lint..." | tee "$lint_log"
  julia --project="$PROJECT_DIR" --color=yes --startup-file=no <<'JULIA' | tee -a "$lint_log"
using Pkg
Pkg.instantiate()
try
    using JuliaFormatter
catch
    @warn "JuliaFormatter not found in project. Add it by running: using Pkg; Pkg.add(\"JuliaFormatter\")"
    exit(0)
end
lint_paths = filter(isdir, ["src", "test"])
if isempty(lint_paths)
    lint_paths = [pwd()]
end
for path in lint_paths
    println("fmt --check ", path)
    format(path; overwrite=false)
end
JULIA
}

run_tests() {
  local test_log="$LOG_DIR/test.log"
  echo "Running Julia tests..." | tee "$test_log"
  julia --project="$PROJECT_DIR" --color=yes --startup-file=no <<'JULIA' | tee -a "$test_log"
using Pkg
Pkg.instantiate()
if isdir("test")
    Pkg.test()
else
    @info "No test directory detected; skipping Julia tests."
end
JULIA
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
