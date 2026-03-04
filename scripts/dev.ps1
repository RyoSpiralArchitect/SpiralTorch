param(
    [Parameter(Position = 0)]
    [string]$Task = "help"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function RunCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$File,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Args
    )

    $pretty = ($Args | ForEach-Object { if ($_ -match "\\s") { '"' + $_ + '"' } else { $_ } }) -join " "
    Write-Host ("> {0} {1}" -f $File, $pretty)
    & $File @Args
    if ($LASTEXITCODE -ne 0) {
        throw ("Command failed (exit {0}): {1} {2}" -f $LASTEXITCODE, $File, $pretty)
    }
}

function ShowHelp {
    Write-Host "SpiralTorch dev helper"
    Write-Host ""
    Write-Host "Usage:"
    Write-Host "  pwsh -File scripts/dev.ps1 <task>"
    Write-Host ""
    Write-Host "Tasks:"
    Write-Host "  fmt         cargo fmt --all"
    Write-Host "  clippy      cargo clippy --workspace --all-targets"
    Write-Host "  core-build  cargo build -p st-core --release"
    Write-Host "  core-test   cargo test  -p st-core --release -- --nocapture"
    Write-Host "  stack       build st-nn / st-spiral-rl / st-rec (release)"
    Write-Host "  wheel       build + install Python wheel (wgpu,logic,kdsl)"
    Write-Host "  docs-check  run docs sanity checks"
}

switch ($Task) {
    "fmt" {
        RunCommand cargo fmt --all
    }
    "clippy" {
        RunCommand cargo clippy --workspace --all-targets
    }
    "core-build" {
        RunCommand cargo build -p st-core --release
    }
    "core-test" {
        RunCommand cargo test -p st-core --release -- --nocapture
    }
    "stack" {
        RunCommand cargo build -p st-nn --release
        RunCommand cargo build -p st-spiral-rl --release
        RunCommand cargo build -p st-rec --release
    }
    "wheel" {
        RunCommand python -m pip install -U pip wheel "maturin>=1,<2"
        RunCommand maturin build -m bindings/st-py/Cargo.toml --release --locked --features wgpu,logic,kdsl
        RunCommand python -m pip install --force-reinstall --no-cache-dir target/wheels/spiraltorch-*.whl
    }
    "docs-check" {
        RunCommand python tools/check_example_gallery.py
        RunCommand python tools/run_readme_python_blocks.py --readme README.md --cwd .
        RunCommand cargo run -p st-bench --bin backend_matrix_md -- --check --doc docs/backend_matrix.md
    }
    default {
        ShowHelp
    }
}

