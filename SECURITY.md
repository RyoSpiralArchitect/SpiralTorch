# Security Policy

We take SpiralTorch's safety seriously. If you discover a security issue,
please contact maintainers privately before disclosing it.

## Supported Versions

SpiralTorch is currently pre-release software. Security fixes are developed on
the `main` branch and will be included in the next tagged release.

## Reporting a Vulnerability

Email: kishkavsesvit@icloud.com

Please include:

- A detailed description of the issue and potential impact.
- Steps to reproduce or proof-of-concept code.
- Suggested mitigations if available.

We will acknowledge receipt within 72 hours and keep you informed as we work on
a resolution. Once a fix is available we will coordinate a disclosure timeline
with you.

## Verifying Official Releases

To help you distinguish authentic SpiralTorch builds from malicious clones or
repackaged binaries, each GitHub release now provides:

- **Sigstore signatures** for every wheel and verification artifact, produced
  via GitHub OIDC and tied to the `Release Wheels` workflow ref.
- **SLSA provenance attestations** covering all wheel artifacts.
- **Cryptographic checksum manifests** (`wheels.sha256` and `wheels.sha512`)
  for quick offline integrity checks.
- **CycloneDX SBOMs** that document the exact dependency graph shipped in each
  wheel, enabling automated diffing against suspicious redistributions.
- **A signed release manifest** (`spiraltorch-<tag>-manifest.json`) that
  captures the digests and sizes of every official asset.
- **An AGPL compliance report** (`spiraltorch-<tag>-license-report.json`) that
  proves every published wheel still embeds the verbatim AGPL-3.0-or-later
  license text from this repository, preventing bad actors from stripping or
  altering the terms during redistribution.
- **A sealed repository manifest and compliance seal**
  (`spiraltorch-repo-license-manifest.json` and
  `spiraltorch-compliance-seal.json`) that bind the release back to the exact
  AGPL-governed Git commit. The seal hashes the manifest, records the canonical
  license digests, and requires downstream mirrors to preserve those files.

### Automated verification

A scheduled workflow (`Verify Official Release Integrity`) re-validates the
latest published release on GitHub every day and whenever a new release is
tagged. The job downloads all assets, checks their hashes against the signed
manifest, and uses `sigstore verify github` to ensure every signature matches
the repository, workflow name, and tag that produced it. Any regression will
fail the workflow to alert maintainers before clones or mirrors can circulate
tampered artifacts.

### Local verification

If you need to audit a release yourself (for example, before mirroring it in a
controlled environment), run:

```bash
pip install sigstore requests
python scripts/security/verify_release.py --tag <release-tag>
```

Provide `--repo <owner/repo>` if you are not running from a cloned copy of the
official repository. The script will:

1. Download the signed manifest, confirm its Sigstore signature, and ensure it
   was produced by the `Release Wheels` workflow on the requested tag.
2. Fetch every asset listed in the manifest and recompute its SHA-256/SHA-512
   digests and file size.
3. Confirm each wheel still contains the canonical AGPL license payload as
   recorded in the signed license provenance report.
4. Validate the compliance seal binds the manifest to the canonical AGPL
   commit, cross-checking its digests against the manifest and license report.
5. Verify the Sigstore signature and certificate pair accompanying each asset.

Treat any mismatched digest, missing signature, or verification failure as a
sign that the artifact may have been tampered with or repackaged. Contact the
maintainers immediately so we can investigate before malicious mirrors can
propagate compromised binaries.

## Locking down repository clones

Bad actors sometimes try to republish the entire SpiralTorch repository with
the AGPL removed or altered. To make that effectively impossible, we ship a
signed manifest of the Git tree itself and require every tracked package to
declare the AGPL-3.0-or-later terms.

- The `Repository License Manifest` workflow runs on every push to `main` and
  on demand. It refuses to execute if the working tree is dirty, verifies that
  the canonical `LICENSE .txt` still hashes to the official digest, checks that
  `NOTICE` explicitly mentions the AGPL, and inspects every `Cargo.toml` and
  `pyproject.toml` to ensure the license metadata still points at
  AGPL-3.0-or-later. Once these invariants pass, it publishes a
  `spiraltorch-repo-license-manifest.json` file and signs it with Sigstore.
- The manifest captures a SHA-256 and SHA-512 digest for every tracked file in
  the repository, along with the license declaration extracted from each Rust
  crate and Python distribution. Any attempt to ship a fork with license text
  removed will change the digests and be immediately detectable.
- Use `scripts/security/verify_repo_clone.py` together with the signed manifest
  **and** its compliance seal to audit a clone:

  ```bash
  pip install sigstore
  sigstore verify github \
    --certificate spiraltorch-repo-license-manifest.json.crt \
    --signature spiraltorch-repo-license-manifest.json.sig \
    --repository SpiralTorch/SpiralTorch \
    --ref refs/heads/main \
    --name "Repository License Manifest" \
    --trigger push \
    spiraltorch-repo-license-manifest.json
  python scripts/security/verify_repo_clone.py \
    --manifest spiraltorch-repo-license-manifest.json \
    --seal spiraltorch-compliance-seal.json \
    --repo-root /path/to/clone
  ```

  If the verification script reports a mismatch, treat the clone as
  compromised—either the AGPL has been stripped, the repository has been
  tampered with, or someone is attempting to sell an unofficial fork that is
  not sealed to the canonical commit.

  The compliance seal requirement forces malicious redistributors to reveal the
  exact commit they are shipping. Because only the official repository can
  obtain a Sigstore certificate for that commit and manifest, altered forks
  cannot mint a matching seal without reintroducing every AGPL obligation.

### Extended clone auditing

The `verify_repo_clone.py` helper now ships with hardened operator tooling to
make redistribution scams and license stripping even less practical:

- **PGP signature validation.** Provide `--manifest-signature` and optionally
  `--seal-signature` together with `--pgp-keyring` to verify the manifest and
  compliance seal against an offline-trusted keyring before parsing them. This
  protects you from forged manifests that were never published by the official
  maintainers.
- **CI enforcement mode.** Add `--ci-mode` (optionally with
  `--check-name=<custom name>`) inside GitHub Actions jobs to emit a rich Check
  Run directly in the Pull Request UI. Any AGPL violation or tampering signal
  fails the check, making it impossible to merge non-compliant contributions
  unnoticed.
- **Human-friendly HTML reports.** Supply `--html-report compliance.html` to
  capture every verification step—including warnings and historical findings—in
  an easy-to-read document for legal, compliance, or management review.
- **Historical tamper scanning.** Pass `--audit-history` (with
  `--history-window` to control the depth) to inspect recent git commits for
  suspicious license deletions, renames, or edits that remove AGPL references.
  The findings are surfaced alongside the main verification output so you can
  respond before a malicious fork grows roots.

Example CI invocation that enables all hardened checks:

```bash
python scripts/security/verify_repo_clone.py \
  --manifest spiraltorch-repo-license-manifest.json \
  --manifest-signature spiraltorch-repo-license-manifest.json.asc \
  --seal spiraltorch-compliance-seal.json \
  --seal-signature spiraltorch-compliance-seal.json.asc \
  --pgp-keyring .github/trust/spiraltorch.gpg \
  --repo-root "$GITHUB_WORKSPACE" \
  --audit-history --history-window 200 \
  --html-report compliance-report.html \
  --ci-mode --check-name "AGPL Compliance Gate"
```

Treat any CI failure, HTML report entry under "Failures", or historical
warning as a release-blocking issue—downstream mirrors and resellers must be
forced to carry the exact AGPL state we publish.
