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
3. Verify the Sigstore signature and certificate pair accompanying each asset.

Treat any mismatched digest, missing signature, or verification failure as a
sign that the artifact may have been tampered with or repackaged. Contact the
maintainers immediately so we can investigate before malicious mirrors can
propagate compromised binaries.
