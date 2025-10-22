# Julia & Go Development Flow Draft

This document captures the draft process for experimenting with future Julia and Go bindings inside SpiralTorch. The flow covers local environment setup, running language-specific checks, and how the CI/CD prototype ties into release preparation.

## 1. Environment setup

### Julia
1. Ensure system prerequisites are present (`curl`, `sudo` on Linux).
2. Run the helper recipe:
   ```bash
   just setup-julia
   ```
   The script installs Julia via `apt` on Ubuntu, Homebrew on macOS, or prints a fallback message with the official installer (`juliaup`).
3. Create or clone a Julia project under `bindings/julia`. The tooling expects a `Project.toml` file at that location.

### Go
1. Ensure you have administrator rights for package managers when needed.
2. Run the helper recipe:
   ```bash
   just setup-go
   ```
   The script installs Go via `apt` on Ubuntu, Homebrew on macOS, or directs you to the official binaries on other platforms.
3. Place your module sources under `bindings/go` with a valid `go.mod` file.

## 2. Linting and testing commands

Two wrapper scripts live in `scripts/` to standardise local runs and log capture:

- `scripts/run_julia_checks.sh [lint|test|all]`
- `scripts/run_go_checks.sh [lint|test|all]`

Both scripts write structured logs to `logs/julia` and `logs/go` respectively. When a language toolchain is missing, the scripts emit actionable hints and exit gracefully.

Typical usage:

```bash
# Julia lint + tests with logs captured under logs/julia/
JULIA_PROJECT_PATH=bindings/julia just julia-lint
JULIA_PROJECT_PATH=bindings/julia just julia-test

# Go lint + tests with logs captured under logs/go/
GO_MODULE_PATH=bindings/go just go-lint
GO_MODULE_PATH=bindings/go just go-test
```

When invoked without arguments the scripts run both lint and tests. They attempt to instantiate dependencies (`Pkg.instantiate()` for Julia, `go test ./...` for Go) and fall back with informational logs when a project directory is absent.

To inspect collected logs:

```bash
ls logs/julia
cat logs/go/test.log
```

## 3. CI workflow draft

The `.github/workflows/julia-go-ci.yml` workflow exercises the scripts on GitHub-hosted runners:

- **Julia job** installs Julia 1.10, runs lint and tests, and uploads `logs/julia` as an artifact.
- **Go job** installs Go 1.22 plus `golangci-lint`, executes lint/tests, and uploads `logs/go`.

Artifacts let maintainers review formatter/test output even when the jobs succeed.

## 4. Release outline

1. Ensure `main` is green across Rust, Julia, and Go workflows.
2. Tag the repository only after Julia and Go logs confirm clean lint/test runs.
3. For Julia bindings, publish the package using the standard registry process after confirming `Pkg.test()`.
4. For Go bindings, push versioned tags and run `go mod tidy` before release candidates.
5. Update release notes to mention the Julia/Go component versions and link the CI artifacts.

This flow is intentionally lightweight; as the bindings mature, replace the placeholder scripts with project-specific tooling.
