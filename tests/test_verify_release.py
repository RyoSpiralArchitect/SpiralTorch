#!/usr/bin/env python3
"""Unit coverage for the official release integrity verifier."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "security" / "verify_release.py"
SPEC = importlib.util.spec_from_file_location("verify_release", SCRIPT)
assert SPEC and SPEC.loader
verify_release = importlib.util.module_from_spec(SPEC)
with mock.patch.dict(sys.modules, {"requests": types.ModuleType("requests")}):
    SPEC.loader.exec_module(verify_release)


class VerifyReleaseTests(unittest.TestCase):
    def test_bundle_verification_is_offline_and_identity_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "artifact.whl"
            artifact.write_bytes(b"wheel")
            verify_release.sigstore_bundle_path(artifact).write_text("{}", encoding="utf-8")

            command = verify_release.sigstore_verify_command(
                artifact,
                "RyoSpiralArchitect/SpiralTorch",
                "refs/tags/v0.4.14",
                "push",
            )

        self.assertIn("--bundle", command)
        self.assertIn("--offline", command)
        self.assertEqual(command[command.index("--repository") + 1], "RyoSpiralArchitect/SpiralTorch")
        self.assertEqual(command[command.index("--ref") + 1], "refs/tags/v0.4.14")
        self.assertEqual(command[command.index("--trigger") + 1], "push")

    def test_legacy_signature_verification_keeps_online_trust_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "artifact.whl"
            artifact.write_bytes(b"wheel")
            certificate, signature = verify_release.legacy_sigstore_paths(artifact)
            certificate.write_text("certificate", encoding="utf-8")
            signature.write_text("signature", encoding="utf-8")

            command = verify_release.sigstore_verify_command(
                artifact,
                "RyoSpiralArchitect/SpiralTorch",
                "refs/tags/v0.4.14",
                "push",
            )

        self.assertIn("--certificate", command)
        self.assertIn("--signature", command)
        self.assertNotIn("--offline", command)


if __name__ == "__main__":
    unittest.main()
