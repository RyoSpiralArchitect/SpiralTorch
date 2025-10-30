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

- **Sigstore signatures** for every wheel, produced via GitHub OIDC so you can
  verify provenance with `cosign verify-blob`.
- **SLSA provenance attestations** covering all wheel artifacts.
- **Cryptographic checksum manifests** (`wheels.sha256` and `wheels.sha512`)
  for quick offline integrity checks.
- **CycloneDX SBOMs** that document the exact dependency graph shipped in each
  wheel, enabling automated diffing against suspicious redistributions.

Download the checksums, SBOM, and attestations directly from the release page
and validate them before installing. If any signature verification fails or an
artifact is missing these protections, treat it as untrusted and contact the
maintainers immediately.
