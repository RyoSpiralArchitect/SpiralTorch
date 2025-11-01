#!/usr/bin/env python3
"""Verify a local repository clone against the signed SpiralTorch license manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Iterable, List, Optional


try:
    from dataclasses import dataclass, field
except ImportError:  # pragma: no cover - Python <3.7 fallback is unused in repo tooling
    dataclass = None  # type: ignore


if dataclass is None:  # pragma: no cover - defensive path for ancient interpreters
    raise SystemExit("Python 3.7+ is required to run the compliance verification tooling.")


from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REQUIRED_LICENSE_TOKEN = "AGPL-3.0-or-later"


SCHEMA = "https://spiraltorch.org/security/compliance-seal/v1"


@dataclass
class VerificationReport:
    checks: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    history_findings: List[str] = field(default_factory=list)

    def add_check(self, message: str) -> None:
        self.checks.append(message)

    def add_failure(self, message: str) -> None:
        self.failures.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def add_history(self, message: str) -> None:
        self.history_findings.append(message)

    @property
    def success(self) -> bool:
        return not self.failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to a trusted spiraltorch-repo-license-manifest.json file.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository clone root to validate. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--require-extra",
        action="append",
        dest="required_paths",
        default=[],
        help="Additional file paths that must exist in the clone. Relative to the repository root.",
    )
    parser.add_argument(
        "--seal",
        type=Path,
        default=None,
        help="Optional compliance seal to enforce commit provenance and manifest digests.",
    )
    parser.add_argument(
        "--manifest-signature",
        type=Path,
        default=None,
        help="Optional detached signature for the manifest file to verify with PGP.",
    )
    parser.add_argument(
        "--seal-signature",
        type=Path,
        default=None,
        help="Optional detached signature for the compliance seal file.",
    )
    parser.add_argument(
        "--pgp-keyring",
        type=Path,
        default=None,
        help="Keyring containing trusted signing keys for PGP verification.",
    )
    parser.add_argument(
        "--ci-mode",
        action="store_true",
        help="Emit GitHub Check API results instead of printing failures to stdout only.",
    )
    parser.add_argument(
        "--check-name",
        default="Verify Repo Clone Compliance",
        help="Override the GitHub Check run name when --ci-mode is enabled.",
    )
    parser.add_argument(
        "--html-report",
        type=Path,
        default=None,
        help="Write a human-readable HTML compliance report to the provided path.",
    )
    parser.add_argument(
        "--audit-history",
        action="store_true",
        help="Scan recent git history for suspicious license tampering events.",
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=50,
        help="Number of recent commits to inspect when --audit-history is set (default: 50).",
    )
    return parser.parse_args()


def digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"Required file is missing: {path}")


def verify_pgp_signature(
    target: Path,
    signature: Path,
    keyring: Optional[Path],
    description: str,
    report: VerificationReport,
) -> None:
    if not signature.is_file():
        raise SystemExit(f"Detached signature not found for {description}: {signature}")
    if keyring and not keyring.exists():
        raise SystemExit(f"PGP keyring not found: {keyring}")

    verifier: Optional[str] = shutil.which("gpgv")
    command: List[str]
    if verifier:
        command = [verifier]
        if keyring:
            command.extend(["--keyring", str(keyring)])
    else:
        verifier = shutil.which("gpg")
        if not verifier:
            raise SystemExit("Neither gpgv nor gpg is available to verify PGP signatures.")
        command = [verifier, "--batch", "--verify"]
        if keyring:
            command.extend(["--keyring", str(keyring)])
    command.extend([str(signature), str(target)])

    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore")
        raise SystemExit(
            f"PGP signature validation failed for {description}. Details:\n{stderr}"
        ) from exc

    output = completed.stderr.decode("utf-8", errors="ignore") or completed.stdout.decode(
        "utf-8", errors="ignore"
    )
    report.add_check(f"Verified {description} signature with {Path(verifier).name}.")
    if output:
        trimmed = "\n".join(line.strip() for line in output.splitlines() if line.strip())
        report.add_check(f"Signature details:\n{trimmed}")


def current_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover - relies on git availability
        raise SystemExit(
            "Unable to resolve HEAD commit for clone; ensure you are inside a git checkout."
        ) from exc
    commit = result.stdout.decode("utf-8").strip()
    if not commit:
        raise SystemExit("Unable to determine HEAD commit for the repository clone.")
    return commit


def audit_history(
    repo_root: Path, canonical_path: Optional[str], history_window: int, report: VerificationReport
) -> None:
    if not canonical_path:
        report.add_warning(
            "Skipping history audit because manifest omitted canonical license path metadata."
        )
        return

    log_command = [
        "git",
        "log",
        f"--max-count={history_window}",
        "--pretty=format:%H%x00%an%x00%ad%x00%s",
        "--name-status",
        "--",
        canonical_path,
    ]
    try:
        result = subprocess.run(
            log_command,
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore")
        report.add_warning(
            "Unable to inspect git history for license tampering. Ensure git is available."
        )
        if stderr.strip():
            report.add_warning(stderr.strip())
        return

    output = result.stdout.decode("utf-8", errors="ignore")
    entries = output.strip().splitlines()
    current_record: Optional[str] = None
    suspicious: List[str] = []

    for line in entries:
        if "\x00" in line:
            current_record = line
            continue
        if not line or current_record is None:
            continue
        status, _, path = line.partition("\t")
        commit_hash = current_record.split("\x00")[0]
        if status.startswith("D"):
            suspicious.append(f"Commit {commit_hash} deleted the canonical license file.")
        elif status.startswith("R"):
            suspicious.append(
                f"Commit {commit_hash} renamed the canonical license file to {path}."
            )

    diff_command = [
        "git",
        "log",
        f"--max-count={history_window}",
        "-p",
        "--",
        canonical_path,
    ]
    diff_result = subprocess.run(
        diff_command,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    diff_output = diff_result.stdout.decode("utf-8", errors="ignore")
    for block in diff_output.split("diff --git"):
        if "@@" not in block:
            continue
        header, _, body = block.partition("@@")
        commit_hash: Optional[str] = None
        for line in header.splitlines():
            if line.startswith("commit "):
                commit_hash = line.split()[1]
                break
        removed_agpl = any(
            line.startswith("-") and REQUIRED_LICENSE_TOKEN not in line for line in body.splitlines()
        )
        if removed_agpl:
            suspicious.append(
                f"Commit {commit_hash or '<unknown>'} removes lines lacking AGPL markers from the canonical license."
            )

    try:
        head_commit = current_commit(repo_root)
    except SystemExit as exc:
        report.add_warning(str(exc))
        head_commit = None
    for finding in suspicious:
        if head_commit and head_commit in finding:
            report.add_warning(
                "HEAD commit contains suspicious license changes. Review immediately."
            )
        report.add_history(finding)

    if suspicious:
        report.add_warning(
            f"Detected {len(suspicious)} historical events that may indicate AGPL tampering."
        )
    else:
        report.add_check(
            f"Reviewed the last {history_window} commits touching {canonical_path}; no tampering detected."
        )


def build_markdown_report(report: VerificationReport) -> str:
    sections: List[str] = []
    if report.checks:
        checks = "\n".join(f"- {line}" for line in report.checks)
        sections.append(f"### ✅ Passed Checks\n{checks}")
    if report.warnings:
        warnings = "\n".join(f"- {line}" for line in report.warnings)
        sections.append(f"### ⚠️ Warnings\n{warnings}")
    if report.history_findings:
        history = "\n".join(f"- {line}" for line in report.history_findings)
        sections.append(f"### 🕰 Historical Findings\n{history}")
    if report.failures:
        failures = "\n".join(f"- {line}" for line in report.failures)
        sections.append(f"### ❌ Failures\n{failures}")
    return "\n\n".join(sections) if sections else "No verification results captured."


def html_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def write_html_report(report: VerificationReport, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    sections = []
    for title, entries in [
        ("Passed Checks", report.checks),
        ("Warnings", report.warnings),
        ("Historical Findings", report.history_findings),
        ("Failures", report.failures),
    ]:
        items = "\n".join(
            f"<li>{html_escape(item)}</li>" for item in entries
        ) or "<li>None</li>"
        sections.append(f"<section><h2>{html_escape(title)}</h2><ul>{items}</ul></section>")

    html = textwrap.dedent(
        f"""
        <!DOCTYPE html>
        <html lang=\"en\">
          <head>
            <meta charset=\"utf-8\" />
            <title>SpiralTorch Compliance Verification Report</title>
            <style>
              body {{ font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.5; }}
              h1 {{ border-bottom: 2px solid #444; padding-bottom: 0.5rem; }}
              section {{ margin-bottom: 1.5rem; }}
              ul {{ list-style: disc inside; }}
            </style>
          </head>
          <body>
            <h1>SpiralTorch Compliance Verification Report</h1>
            <p>Generated by <code>verify_repo_clone.py</code> to capture machine-verifiable AGPL compliance guarantees.</p>
            {''.join(sections)}
          </body>
        </html>
        """
    ).strip()
    destination.write_text(html, encoding="utf-8")


def emit_check_run(args: argparse.Namespace, report: VerificationReport, success_message: str) -> None:
    repository = os.environ.get("GITHUB_REPOSITORY")
    token = os.environ.get("GITHUB_TOKEN")
    sha = os.environ.get("GITHUB_SHA")
    if not sha:
        try:
            sha = current_commit(args.repo_root)
        except SystemExit as exc:
            raise SystemExit(f"--ci-mode could not determine HEAD commit: {exc}") from exc

    if not repository or not token:
        raise SystemExit(
            "--ci-mode requested but GITHUB_REPOSITORY/GITHUB_TOKEN were not provided in the environment."
        )

    url = f"https://api.github.com/repos/{repository}/check-runs"
    body = {
        "name": args.check_name,
        "head_sha": sha,
        "status": "completed",
        "conclusion": "success" if report.success else "failure",
        "output": {
            "title": args.check_name,
            "summary": success_message if report.success else "Compliance violations detected.",
            "text": build_markdown_report(report),
        },
    }

    data = json.dumps(body).encode("utf-8")
    request = Request(url, data=data, method="POST")
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("User-Agent", "spiraltorch-compliance-bot")

    try:
        with urlopen(request) as response:
            response.read()
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore")
        raise SystemExit(
            f"Failed to create GitHub Check run ({exc.code}): {error_body}"
        ) from exc
    except URLError as exc:
        raise SystemExit(f"Unable to reach GitHub API: {exc.reason}") from exc


def print_summary(report: VerificationReport) -> None:
    if report.checks:
        print("✅ Completed checks:")
        for entry in report.checks:
            print(f" - {entry}")
    if report.warnings:
        print("⚠️ Warnings:")
        for entry in report.warnings:
            print(f" - {entry}")
    if report.history_findings:
        print("🕰 Historical findings:")
        for entry in report.history_findings:
            print(f" - {entry}")
    if report.failures:
        print("❌ Failures:")
        for entry in report.failures:
            print(f" - {entry}")


def validate_seal(
    seal_data: dict,
    seal_path: Path,
    manifest_path: Path,
    repo_root: Path,
    report: VerificationReport,
) -> bool:
    ok = True
    if seal_data.get("schema") != SCHEMA:
        report.add_failure(f"Unsupported compliance seal schema in {seal_path}")
        return False

    manifest_info = seal_data.get("manifest") or {}
    expected_sha256 = manifest_info.get("sha256")
    expected_sha512 = manifest_info.get("sha512")
    if not expected_sha256 or not expected_sha512:
        report.add_failure("Compliance seal is missing manifest digests.")
        ok = False
    
    actual_sha256 = digest(manifest_path, "sha256")
    actual_sha512 = digest(manifest_path, "sha512")
    if actual_sha256 != expected_sha256 or actual_sha512 != expected_sha512:
        report.add_failure("Compliance seal does not match the supplied manifest file.")
        ok = False

    canonical = seal_data.get("canonical_license") or {}
    for key in ["path", "sha256", "sha512"]:
        if not canonical.get(key):
            report.add_failure("Compliance seal is missing canonical license metadata.")
            ok = False

    clause = seal_data.get("agpl_clause", "")
    if REQUIRED_LICENSE_TOKEN not in clause:
        report.add_failure("Compliance seal clause fails to reference AGPL obligations.")
        ok = False

    try:
        repo_head = current_commit(repo_root)
    except SystemExit as exc:
        report.add_failure(str(exc))
        return False
    seal_commit = seal_data.get("commit")
    if not seal_commit:
        report.add_failure("Compliance seal omitted the source commit hash.")
        ok = False
    elif repo_head != seal_commit:
        report.add_failure(
            "Repository clone is not checked out at the sealed commit. Ensure you are inspecting the official AGPL state before use."
        )
        ok = False

    for required in seal_data.get("required_files", []) or []:
        path = repo_root / required
        if not path.exists():
            report.add_failure(f"Compliance seal required file missing from clone: {required}")
        else:
            report.add_check(f"Compliance seal required file present: {required}")

    if ok and not report.failures:
        report.add_check(
            "Compliance seal validated: manifest digests, canonical license metadata, and commit binding verified."
        )
    return ok


def validate_clone(
    manifest: dict,
    repo_root: Path,
    required_paths: Iterable[str],
    report: VerificationReport,
) -> None:
    files = manifest.get("files") or []
    if not files:
        raise SystemExit("Manifest contained no file entries to compare.")

    repo_root = repo_root.resolve()

    for entry in files:
        relative = entry.get("path")
        if not relative:
            report.add_failure("Manifest entry missing path field.")
            continue
        sha256 = entry.get("sha256")
        sha512 = entry.get("sha512")
        size = entry.get("size")
        if not all([sha256, sha512, size is not None]):
            report.add_failure(f"Manifest entry for {relative} is incomplete.")
            continue

        local_path = repo_root / relative
        if not local_path.is_file():
            report.add_failure(f"Missing tracked file: {relative}")
            continue

        local_sha256 = digest(local_path, "sha256")
        local_sha512 = digest(local_path, "sha512")
        if local_sha256 != sha256:
            report.add_failure(f"SHA256 mismatch for {relative}")
        if local_sha512 != sha512:
            report.add_failure(f"SHA512 mismatch for {relative}")

        local_size = local_path.stat().st_size
        if local_size != size:
            report.add_failure(
                f"Size mismatch for {relative}: expected {size}, found {local_size}"
            )

    canonical = manifest.get("canonical_license") or {}
    canonical_path = canonical.get("path")
    canonical_sha256 = canonical.get("sha256")
    canonical_sha512 = canonical.get("sha512")
    if not canonical_path or not canonical_sha256 or not canonical_sha512:
        report.add_failure("Manifest did not include canonical license metadata.")
    else:
        license_path = repo_root / canonical_path
        if not license_path.is_file():
            report.add_failure(f"Canonical license missing from clone: {canonical_path}")
        else:
            if digest(license_path, "sha256") != canonical_sha256:
                report.add_failure("Canonical license SHA256 mismatch.")
            if digest(license_path, "sha512") != canonical_sha512:
                report.add_failure("Canonical license SHA512 mismatch.")
            license_text = license_path.read_text(encoding="utf-8", errors="ignore")
            if REQUIRED_LICENSE_TOKEN not in license_text:
                report.add_failure(
                    "Canonical license file does not reference the AGPL obligations."
                )
            else:
                report.add_check("Canonical AGPL license present and matches recorded digests.")

    compliance = manifest.get("compliance") or {}
    for section, entries in compliance.items():
        for entry in entries or []:
            license_expression = entry.get("license", "")
            if REQUIRED_LICENSE_TOKEN not in license_expression:
                report.add_failure(
                    f"{section} manifest {entry.get('manifest', '<unknown>')} missing AGPL expression: {license_expression}"
                )
            manifest_path = entry.get("manifest")
            if manifest_path:
                local_manifest = repo_root / manifest_path
                if not local_manifest.is_file():
                    report.add_failure(f"Compliance manifest missing from clone: {manifest_path}")
                else:
                    report.add_check(f"Found compliance manifest: {manifest_path}")

    for extra in required_paths:
        target = repo_root / extra
        if not target.exists():
            report.add_failure(f"Required file is missing: {extra}")
        else:
            report.add_check(f"Verified required file: {extra}")


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    report = VerificationReport()

    manifest_signature = args.manifest_signature.resolve() if args.manifest_signature else None
    seal_signature = args.seal_signature.resolve() if args.seal_signature else None
    keyring = args.pgp_keyring.resolve() if args.pgp_keyring else None

    if manifest_signature:
        verify_pgp_signature(
            manifest_path,
            manifest_signature,
            keyring,
            "repository license manifest",
            report,
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    seal_data = None
    if args.seal:
        seal_path = args.seal.resolve()
        if not seal_path.is_file():
            raise SystemExit(f"Compliance seal not found: {seal_path}")
        seal_data = json.loads(seal_path.read_text(encoding="utf-8"))
        if seal_signature:
            verify_pgp_signature(
                seal_path,
                seal_signature,
                keyring,
                "compliance seal",
                report,
            )
        validate_seal(seal_data, seal_path, manifest_path, args.repo_root, report)

    validate_clone(manifest, args.repo_root, args.required_paths, report)

    if args.audit_history:
        canonical_path = (manifest.get("canonical_license") or {}).get("path")
        audit_history(args.repo_root, canonical_path, args.history_window, report)

    success_message = "Repository clone matches the signed manifest and preserves AGPL declarations."

    if args.html_report:
        write_html_report(report, args.html_report.resolve())

    if args.ci_mode:
        emit_check_run(args, report, success_message)

    print_summary(report)

    if report.success:
        print(success_message)
    else:
        raise SystemExit("Repository clone failed compliance verification. See failures above.")


if __name__ == "__main__":
    main()
