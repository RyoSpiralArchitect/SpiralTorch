# SpiralTorch Release Operations

This runbook keeps the PyPI path explicit and auditable. The safe default is a
GitHub Actions dry-run: it validates the signed release wheels and PyPI state
without uploading.

## Release Wheel Workflows

- Manual wheel artifact build: `.github/workflows/wheels.yml`
- Official release build + attached assets: `.github/workflows/release_wheels.yml`
- Publish existing signed GitHub Release wheels: `.github/workflows/publish_pypi_from_release.yml`
- Release readiness summary: `scripts/release_status.py`
- Safe PyPI token secret setup: `scripts/configure_pypi_token_secret.py`
- Safe publish workflow runner: `scripts/run_pypi_publish_from_release.py`
- Safe manual PyPI publish helper: `scripts/publish_pypi_wheels.py`
- Published-wheel digest verifier: `scripts/security/verify_pypi_release.py`

Both PyPI workflows call the same manifest-backed wheel validator before any
token or trusted-publisher upload. Official release builds also execute all HF
and Z-Space console entrypoints on Linux, macOS, and Windows after installing
each wheel, so a platform-specific missing runtime payload blocks publication.
They also execute the installed wheel through the Rust-owned runtime protocol
catalog plus generation-evidence, token-periodicity, repetition-plan, and
blinded semantic-review lifecycles before any artifact can be uploaded. The
stochastic Schrodinger forward/replay/VJP lifecycle is covered by the same
installed-wheel smoke, including complex-state forward and joint VJP replay.
Legacy/versioned DLPack transfers, capsule single-use, explicit copies, and
copy-on-write mutation are also exercised using the installed native wheel.
Protected autograd snapshots, a complete 600-step nonlinear learning fixture,
atomic SGD failure/ownership checks, a 300-step multiclass logits-loss fixture,
and a 400-step affine LayerNorm learning fixture run
on every installed release wheel before upload. This is a mechanics gate,
not a claim about LLM fine-tuning quality.
Catalog v4 records every normal-admission profile plus
Rust-owned byte/node/depth limits
for serialized Python/WASM surfaces; typed Rust admission has no serialized
budget. Browser object helpers are trusted-local convenience transports and are
not release evidence for hostile-input safety.
Automatic releases require the pushed tag to equal `v<package-version>`
exactly. Manual recovery with a non-empty `release_tag` also requires
`checkout_ref` to name that same immutable tag, keeping wheel bytes,
compliance manifests, signatures, and publication helpers on one source ref.
Leave both inputs empty for a build-only branch preflight.

## Common Variables

```bash
VERSION=0.4.26
TAG="v${VERSION}"
DIST="/tmp/spiraltorch-${VERSION}-dist"
```

## Readiness Snapshot

Run this before any publish attempt. It does not print secret values; the
explicit flag also avoids reading the clipboard.

```bash
python scripts/release_status.py \
  --version "$VERSION" \
  --release-tag "$TAG" \
  --expected-wheels 3 \
  --no-clipboard
```

Expected pre-publish shape for `0.4.26` is:

```text
local_versions ... consistent=yes
github_release ... ready=yes wheels=3/3 wheels_sha256=yes
pypi ... published=no
next_action: python scripts/configure_pypi_token_secret.py --token-source prompt OR configure PyPI Trusted Publishing
```

Current helpers also print concrete resume commands:

```text
token_secret_setup: python scripts/configure_pypi_token_secret.py --token-source prompt
publish_token_workflow: gh workflow run publish_pypi_from_release.yml --ref main -f release_tag=v0.4.26 -f expected_wheels=3 -f publish_method=token -f skip_existing=true
publish_trusted_workflow: gh workflow run publish_pypi_from_release.yml --ref main -f release_tag=v0.4.26 -f expected_wheels=3 -f publish_method=trusted -f skip_existing=true
trusted_publisher sub=repo:RyoSpiralArchitect/SpiralTorch:environment:pypi workflow_ref=RyoSpiralArchitect/SpiralTorch/.github/workflows/publish_pypi_from_release.yml@refs/heads/main environment=pypi
next_action: python scripts/configure_pypi_token_secret.py --token-source prompt OR configure PyPI Trusted Publishing
```

## GitHub Actions Dry-Run

This is safe to run repeatedly. The workflow default is `publish_method=dry-run`,
and the PyPI upload and post-upload smoke steps are skipped in that mode.

```bash
python scripts/run_pypi_publish_from_release.py \
  --version "$VERSION" \
  --publish-method dry-run \
  --watch
```

The runner preflights local/release/PyPI state first, then dispatches and
optionally watches the workflow. If you only want to inspect the exact workflow
command without dispatching it, add `--print-only`.

The equivalent raw workflow command is:

```bash
gh workflow run publish_pypi_from_release.yml \
  --ref main \
  -f release_tag="$TAG" \
  -f expected_wheels=3 \
  -f publish_method=dry-run \
  -f skip_existing=true
```

## Token Secret Setup

Prefer the `pypi` environment secret so the credential scope matches the
workflow environment. The helper does not echo the token and passes it to
`gh secret set` through stdin, not through shell history.

```bash
python scripts/configure_pypi_token_secret.py --token-source prompt
```

If the token secret is intentionally repo-wide instead of environment-scoped,
omit `--env pypi`. If the workflow fails with
`publish_method=token requires a PYPI_API_TOKEN`, the selected GitHub
environment cannot see that secret yet.

If the active shell/agent cannot accept an interactive hidden prompt, use a
stdin handoff instead. This keeps the token out of stdout, shell history, and
process arguments while still feeding `gh secret set` through stdin.

```bash
(
  old_stty=$(stty -g)
  trap 'stty "$old_stty"; unset PYPI_TOKEN' EXIT
  printf 'PyPI token for spiraltorch (hidden): '
  stty -echo
  IFS= read -r PYPI_TOKEN
  stty "$old_stty"
  printf '\n'
  printf '%s' "$PYPI_TOKEN" | python scripts/configure_pypi_token_secret.py --token-source stdin
)
```

For non-interactive local automation, use `--token-source env --token-env
PYPI_API_TOKEN` or pipe another secret manager into `--token-source stdin`.
Use `--dry-run` to validate token shape and secret target without storing it.

## Publish Signed Release Wheels

The preferred PyPI path reuses the already-signed GitHub Release wheels.

```bash
python scripts/run_pypi_publish_from_release.py \
  --version "$VERSION" \
  --publish-method token \
  --watch
```

The runner refuses `--publish-method token` until the `pypi` environment can see
`PYPI_API_TOKEN`, so a missing secret fails before dispatching an upload run.

The equivalent raw workflow command is:

```bash
gh workflow run publish_pypi_from_release.yml \
  --ref main \
  -f release_tag="$TAG" \
  -f expected_wheels=3 \
  -f publish_method=token \
  -f skip_existing=true
```

Use `publish_method=trusted` only after PyPI Trusted Publishing is configured
for the matching workflow and environment.

## Verify Without Republishing

An upload can finish before pip's index reflects the version. Release JSON and
the [Simple API](https://packaging.python.org/en/latest/specifications/simple-repository-api/)
are checked independently: every expected wheel name and SHA-256 must match.
The readiness verifier uses a monotonic retry budget, caps socket timeouts by
the remaining budget, and retries only temporary availability failures.
Malformed responses, unexpected files, mismatched hashes, yanked release wheels,
authentication failures and TLS certificate failures remain terminal.

The normal publish workflow calls a separate read-only verification job after
upload. To recover from a post-upload check failure, verify the existing release
without giving the job a PyPI token or upload authority:

```bash
gh workflow run verify_pypi_release.yml \
  --ref main \
  -f release_tag="$TAG" \
  -f expected_wheels=3 \
  -f require_latest=true
```

Omit `require_latest=true` when intentionally checking a historical release.
The job waits for the index, writes a version/hash-pinned requirements file,
installs from PyPI with `--require-hashes` in an isolated target, and verifies
the imported native package's path and version. A new index entry cannot
substitute a wheel outside the approved release hashes.
It neither rebuilds nor replaces release assets. A failed import remains a
failure; it is not treated as publication lag.

The availability/hash check can also run locally, without installing anything:

```bash
python scripts/security/verify_pypi_release.py \
  --version "$VERSION" \
  --release-tag "$TAG" \
  --expected-wheels 3 \
  --require-latest \
  --require-simple-index \
  --timeout 240
```

Do not rerun the upload merely because an index-readiness or import check failed.
Keep the failed run as evidence and use the verification-only workflow above.

## Local Dry-Run Or Emergency Manual Upload

Download the release payload:

```bash
mkdir -p "$DIST"
gh release download "$TAG" \
  --dir "$DIST" \
  --clobber \
  --pattern 'spiraltorch-*.whl' \
  --pattern 'wheels.sha256'
```

Validate wheels, release checksums, PyPI state, and token shape without
uploading. Use `--token-source none` when you only want wheel/release validation
without credential readiness.

```bash
python scripts/publish_pypi_wheels.py \
  --dist "$DIST" \
  --expected-version "$VERSION" \
  --github-release-tag "$TAG" \
  --token-source prompt \
  --dry-run
```

Real local upload reads a `pypi-...` token from a hidden prompt, uploads with
twine, verifies PyPI wheel digests, then installs/import-smokes the release.

```bash
python scripts/publish_pypi_wheels.py \
  --dist "$DIST" \
  --expected-version "$VERSION" \
  --github-release-tag "$TAG" \
  --token-source prompt \
  --skip-existing
```

If a local multi-wheel upload stalls on a slow uplink, upload one wheel at a
time from a clean twine venv. Keep `--skip-existing` so interrupted retries are
safe after PyPI accepts an earlier file. `read -s` keeps the token out of stdout
and shell history.

```bash
read -rsp "PyPI token for spiraltorch (hidden): " TWINE_PASSWORD
echo
for wheel in "$DIST"/spiraltorch-"$VERSION"-*.whl; do
  TWINE_USERNAME=__token__ TWINE_PASSWORD="$TWINE_PASSWORD" \
    python -m twine upload --non-interactive --skip-existing --disable-progress-bar "$wheel"
done
unset TWINE_PASSWORD
```

## Rebuild Or Recovery

Signed GitHub Release recovery runs the fixed workflow from `main`, rebuilds
wheels from the release tag, regenerates manifest/Sigstore bundles, and
overwrites the assets on that tag's release.

```bash
gh workflow run release_wheels.yml \
  --ref main \
  -f release_tag="$TAG" \
  -f checkout_ref="$TAG" \
  -f publish_pypi=false
```

If you intentionally publish from a rebuild, choose the auth path explicitly.
`publish_pypi=true` requires `release_tag`; after upload, CI verifies PyPI wheel
digests against the GitHub Release `wheels.sha256` emitted by the attach job.

```bash
gh workflow run release_wheels.yml \
  --ref main \
  -f release_tag="$TAG" \
  -f checkout_ref="$TAG" \
  -f publish_pypi=true \
  -f pypi_publish_method=token
```

Re-run integrity verification for the recovered release:

```bash
gh workflow run verify-release.yml --ref main -f release_tag="$TAG"
```

Confirm published PyPI wheels are byte-identical to the GitHub Release wheel
manifest. Add `--require-latest` when publishing the current release. The
verifier polls both the version-specific wheel set and PyPI's latest-version
index for up to `--timeout` seconds, covering the short propagation window
where uploaded wheels are visible before the project index advances.

```bash
python scripts/security/verify_pypi_release.py \
  --version "$VERSION" \
  --release-tag "$TAG" \
  --expected-wheels 3 \
  --require-latest
```

## Trusted Publishing

Trusted publishing is intentionally explicit. For PyPI OIDC, configure the PyPI
publisher for project `spiraltorch` with:

- Owner: `RyoSpiralArchitect`
- Repository: `SpiralTorch`
- Environment: `pypi`
- Workflow file: `publish_pypi_from_release.yml` or `release_wheels.yml`

For the release-wheel reuse workflow, the expected OIDC shape is:

```text
sub=repo:RyoSpiralArchitect/SpiralTorch:environment:pypi
repository=RyoSpiralArchitect/SpiralTorch
workflow_ref=RyoSpiralArchitect/SpiralTorch/.github/workflows/publish_pypi_from_release.yml@refs/heads/main
environment=pypi
```

If the trusted path fails with `invalid-publisher`, the PyPI-side publisher is
missing or one of those fields does not match. Without that PyPI-side publisher,
select `token` and provide `PYPI_API_TOKEN` as a GitHub secret, or use the local
helper with `--token-source prompt` when `pbpaste` is not visible from the
current shell. `--token-source env --token-env PYPI_API_TOKEN` is also available
for non-interactive local automation.
