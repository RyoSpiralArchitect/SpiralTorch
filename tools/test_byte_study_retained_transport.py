import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import byte_study_retained_transport as transport  # noqa: E402


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def packet(self, change=None):
        path = self.root / "packet.tar.gz"
        with tarfile.open(path, "w:gz") as bundle:
            for name in sorted(transport.PACKET_NAMES):
                info = tarfile.TarInfo(name)
                info.size = 2
                if change:
                    change(info)
                bundle.addfile(info, io.BytesIO(b"{}"))
        return path

    def unpack_checked(self, path, target=None):
        with patch.object(transport, "PACKET_SHA256", transport.digest(path)):
            transport.unpack(path, target or self.root / "unpacked")

    def test_exact_members_and_private_modes(self):
        self.unpack_checked(self.packet())
        root = self.root / "unpacked"
        self.assertEqual(root.stat().st_mode & 0o777, 0o700)
        self.assertEqual({str(p.relative_to(root)) for p in root.rglob("*.json")},
                         transport.PACKET_NAMES)
        for path in root.rglob("*.json"):
            self.assertEqual(path.read_bytes(), b"{}")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_changed_packet_rejected_before_output(self):
        with self.assertRaisesRegex(ValueError, "identity"):
            transport.unpack(self.packet(), self.root / "out")
        self.assertFalse((self.root / "out").exists())

    def test_traversal_rejected(self):
        path = self.packet(lambda info: setattr(info, "name", "../outside"))
        with self.assertRaises(ValueError):
            self.unpack_checked(path)
        self.assertFalse((self.root / "unpacked").exists())

    def test_duplicate_members_rejected(self):
        path = self.packet(lambda info: setattr(info, "name", "version-9/request.json"))
        with self.assertRaises(ValueError):
            self.unpack_checked(path)

    def test_link_rejected(self):
        def link(info):
            info.type, info.linkname = tarfile.SYMTYPE, "/tmp/untrusted"
        with self.assertRaises(ValueError):
            self.unpack_checked(self.packet(link))

    def test_existing_destination_not_overwritten(self):
        output = self.root / "existing"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            self.unpack_checked(self.packet(), output)

    def test_pack_includes_complete_hashed_evidence(self):
        evidence = self.root / "evidence"
        evidence.mkdir()
        (evidence / "partial.json").write_bytes(b'{"finished": false}')
        output = self.root / "result.tar.gz"
        manifest = transport.pack(evidence, output)
        self.assertFalse(manifest["release_ready"])
        with tarfile.open(output, "r:gz") as bundle:
            self.assertEqual(set(bundle.getnames()), {
                transport.MANIFEST_NAME, "partial.json"})
            saved = json.loads(bundle.extractfile(transport.MANIFEST_NAME).read())
            raw = bundle.extractfile("partial.json").read()
        self.assertEqual(saved, manifest)
        self.assertEqual(manifest["files"]["partial.json"]["sha256"],
                         hashlib.sha256(raw).hexdigest())
        self.assertEqual(output.stat().st_mode & 0o777, 0o600)
        with self.assertRaises(FileExistsError):
            transport.pack(evidence, output)

    def test_pack_rejects_links_and_self_inclusion(self):
        evidence = self.root / "evidence"
        evidence.mkdir()
        (evidence / "link").symlink_to("/tmp")
        with self.assertRaises(ValueError):
            transport.pack(evidence, self.root / "out.tar.gz")
        with self.assertRaises(ValueError):
            transport.pack(evidence, evidence / "out.tar.gz")


class HostedWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.workflow = (Path(__file__).resolve().parents[1]
                         / ".github/workflows/byte_study_replay.yml").read_text()

    def test_manual_bounded_nonpublishing_single_invocation(self):
        self.assertIn("on:\n  workflow_dispatch:", self.workflow)
        for forbidden in ("push:", "pull_request:", "schedule:", "contents: write",
                          "id-token:", "twine", "publish_pypi", "continue-on-error",
                          "gh release", "git push"):
            self.assertNotIn(forbidden, self.workflow)
        self.assertEqual(self.workflow.count("--execute-once"), 1)
        self.assertIn('test "$GITHUB_RUN_ATTEMPT" = "1"', self.workflow)
        self.assertIn('test "$EXPECTED_HARNESS" = "$GITHUB_SHA"', self.workflow)
        self.assertIn("refs/heads/spiralreality/retained-hosted-once-20261024",
                      self.workflow)
        self.assertIn("timeout-minutes: 180", self.workflow)
        self.assertIn("--timeout-seconds 7200", self.workflow)
        self.assertIn("cancel-in-progress: false", self.workflow)
        self.assertIn("e053c314d54be06b346e30e5ea1a065d7f3fdc90", self.workflow)
        uses = re.findall(r"uses: (\S+)", self.workflow)
        self.assertTrue(uses)
        for action in uses:
            self.assertRegex(action, r"^[\w/-]+@[0-9a-f]{40}$")

    def test_only_ciphertext_uploaded_and_only_input_key_transferred(self):
        self.assertEqual(re.findall(r"secrets\.([A-Z0-9_]+)", self.workflow),
                         ["ST_RETAINED_20261024_INPUT_IDENTITY"])
        self.assertIn("unset INPUT_IDENTITY", self.workflow)
        self.assertIn("trap 'rm -f \"$input_key\"' EXIT", self.workflow)
        upload = self.workflow.split("- name: Upload ciphertext only", 1)[1]
        self.assertIn("if: always()", upload)
        paths = upload.split("path: |\n", 1)[1].split("          if-no-files-found:", 1)[0]
        self.assertEqual([line.strip() for line in paths.splitlines()], [
            "${{ runner.temp }}/retained-20261024/public/evidence.tar.gz.age",
            "${{ runner.temp }}/retained-20261024/public/SHA256SUMS",
        ])
        self.assertNotIn("result-key", self.workflow)
        for name in ("build.log", "input-preflight.log", "launch.log"):
            self.assertIn(f'> "$EVIDENCE_ROOT/{name}" 2>&1', self.workflow)

    def test_legacy_workflow_contract_is_tested_at_unchanged_native_pin(self):
        self.assertIn("../native-source/tools/test_run_byte_study_replay.py",
                      self.workflow)
        self.assertIn("tools/test_byte_study_retained_transport.py", self.workflow)
        self.assertIn("tools/test_run_byte_study_retained.py", self.workflow)

    def test_runner_paths_are_set_after_runner_assignment(self):
        job_settings = self.workflow.split("    steps:", 1)[0]
        self.assertNotIn("runner.temp", job_settings)
        self.assertIn('TRANSPORT_ROOT="$RUNNER_TEMP/retained-20261024"', self.workflow)
        self.assertIn('>> "$GITHUB_ENV"', self.workflow)


if __name__ == "__main__":
    unittest.main()
