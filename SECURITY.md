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
4. Verify the Sigstore signature and certificate pair accompanying each asset.

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
- Use `scripts/security/verify_repo_clone.py` together with a trusted copy of
  the signed manifest to audit a clone:

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
    --repo-root /path/to/clone
  ```

  If the verification script reports a mismatch, treat the clone as
  compromised—either the AGPL has been stripped, or the repository has been
  tampered with in a way that breaks the signed manifest.
