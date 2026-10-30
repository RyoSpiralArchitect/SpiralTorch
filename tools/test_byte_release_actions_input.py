"""Synthetic offline checks; no bridge permission, network or native imports."""
import copy
import hashlib
import io
from pathlib import Path
import stat
import struct
import sys
import tempfile
import warnings
import zipfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import byte_release_actions_input as a
from test_byte_release_transport_v2 import Offline

PAYLOAD = b"synthetic opaque bytes; encryption is not claimed by this fixture"


def zip_bytes(entries=None, compression=zipfile.ZIP_STORED, comment=b""):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=compression) as archive:
        for name, data in entries or [("input.age", PAYLOAD)]:
            archive.writestr(name, data)
        archive.comment = comment
    return stream.getvalue()


def records(raw):
    pin = dict(schema=a.SCHEMA, repository=a.t.REPO, repository_id=101,
               bridge=dict(run_id=202, run_attempt=1, head_sha="a" * 40,
                           branch=a.BRIDGE_PREFIX + "synthetic", workflow_id=303,
                           workflow_path=".github/workflows/" + a.t.WORKFLOW),
               artifact=dict(id=404, name="opaque-input", bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()),
               ciphertext=dict(name="input.age", bytes=len(PAYLOAD), sha256=hashlib.sha256(PAYLOAD).hexdigest()),
               release_ready=False)
    bridge, outer = pin["bridge"], pin["artifact"]
    run = dict(id=202, run_attempt=1, head_sha=bridge["head_sha"], head_branch=bridge["branch"],
               workflow_id=303, path=bridge["workflow_path"], event="workflow_dispatch",
               status="completed", conclusion="success", repository=dict(id=101, full_name=a.t.REPO),
               head_repository=dict(id=101, full_name=a.t.REPO))
    artifact = dict(id=404, name=outer["name"], size_in_bytes=outer["bytes"],
                    digest="sha256:" + outer["sha256"], expired=False,
                    workflow_run=dict(id=202, repository_id=101, head_repository_id=101,
                                      head_sha=bridge["head_sha"], head_branch=bridge["branch"]))
    return pin, run, artifact


def zip64_override(raw):
    fields = struct.unpack("<4s4H2IH", raw[-22:])
    prefix = raw[:-22]
    extended = struct.pack("<4sQ2H2I4Q", b"PK\x06\x06", 44, 45, 45, 0, 0,
                           fields[3], fields[4], fields[5], fields[6])
    locator = struct.pack("<4sIQI", b"PK\x06\x07", 0, len(prefix), 1)
    # The classic record looks bounded, but ZipFile follows the preceding locator.
    fake_end = struct.pack("<4s4H2IH", b"PK\x05\x06", 0, 0, 1, 1, 46,
                           len(prefix) + len(extended) + len(locator) - 46, 0)
    return prefix + extended + locator + fake_end


class ActionsInputOffline(Offline):
    def test_strict_pin_identity_schema_and_opaque_names(self):
        pin, _run, _artifact = records(zip_bytes())
        self.assertEqual(a.validate_pin(pin), pin)
        edits = [((), "repository_id", True), ((), "repository", "another/repository"),
                 ((), "release_ready", 0), ((), "url", "https://example.invalid/?private=fake"),
                 (("bridge",), "run_id", 202.0), (("bridge",), "run_attempt", True),
                 (("bridge",), "run_attempt", 2), (("bridge",), "workflow_id", 0),
                 (("bridge",), "head_sha", "not-a-commit"),
                 (("bridge",), "branch", a.t.branch(1)), (("bridge",), "branch", a.BRIDGE_PREFIX + "../escape"),
                 (("bridge",), "workflow_path", ".github/workflows/other.yml"),
                 (("artifact",), "id", 2**63), (("artifact",), "bytes", True),
                 (("artifact",), "bytes", a.MAX_ARCHIVE + 1), (("artifact",), "sha256", "f" * 63),
                 (("artifact",), "name", "../opaque"), (("artifact",), "name", "opaque.lock"),
                 (("ciphertext",), "name", "input-key.txt"), (("ciphertext",), "name", "nested/input.age")]
        for parents, key, value in edits:
            changed = copy.deepcopy(pin)
            row = changed
            for parent in parents:
                row = row[parent]
            row[key] = value
            with self.subTest(parents=parents, key=key, value=value), self.assertRaises(ValueError):
                a.validate_pin(changed)

    def test_metadata_binds_completed_success_first_attempt_and_repository(self):
        pin, run, artifact = records(zip_bytes())
        self.assertFalse(a.verify_metadata(pin, run, artifact)["release_ready"])
        for key, value in (("id", 203), ("run_attempt", 2), ("run_attempt", True),
                           ("head_sha", "b" * 40), ("head_branch", "main"),
                           ("workflow_id", 303.0), ("path", "other.yml"), ("event", "push"),
                           ("status", "in_progress"), ("conclusion", "failure"),
                           ("conclusion", "cancelled"), ("conclusion", None)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                a.verify_metadata(pin, dict(run, **{key: value}), artifact)
        for key in ("repository", "head_repository"):
            for value in (None, {}, dict(id=True, full_name=a.t.REPO), dict(id=101, full_name="fork/repo")):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    a.verify_metadata(pin, dict(run, **{key: value}), artifact)

    def test_artifact_identity_expiry_and_run_relation_are_not_replaceable(self):
        pin, run, artifact = records(zip_bytes())
        for key, value in (("id", 405), ("id", 404.0), ("name", "other-input"),
                           ("size_in_bytes", pin["artifact"]["bytes"] + 1), ("digest", "sha256:" + "0" * 64),
                           ("expired", True), ("expired", 0), ("workflow_run", None)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                a.verify_metadata(pin, run, dict(artifact, **{key: value}))
        for key, value in (("id", 203), ("id", 202.0), ("repository_id", 102),
                           ("head_repository_id", 102), ("head_sha", "b" * 40), ("head_branch", "main")):
            changed = copy.deepcopy(artifact)
            changed["workflow_run"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                a.verify_metadata(pin, run, changed)

    def test_api_urls_are_not_followed_or_copied_to_receipts(self):
        pin, run, artifact = records(zip_bytes())
        artifact.update(url="https://untrusted.invalid/?credential=fake",
                        archive_download_url="https://untrusted.invalid/?credential=fake")
        result = a.verify_metadata(pin, run, artifact)
        self.assertNotIn("credential", a.t.c.encoded(result).decode())

    def test_stored_and_deflated_archives_preserve_exact_ciphertext_once(self):
        for compression in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            with self.subTest(compression=compression), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                raw = zip_bytes(compression=compression)
                pin, run, artifact = records(raw)
                archive, output = root / "archive.zip", root / "input.age"
                archive.write_bytes(raw)
                result = a.unpack_input(pin, run, artifact, archive, output)
                self.assertEqual(output.read_bytes(), PAYLOAD)
                self.assertEqual(output.stat().st_mode & 0o777, 0o600)
                self.assertEqual(result["status"], "verified_pinned_ciphertext")
                self.assertFalse(result["release_ready"])
                with self.assertRaises(FileExistsError):
                    a.unpack_input(pin, run, artifact, archive, output)

    def test_failed_provenance_or_changed_archive_never_opens_output(self):
        raw = zip_bytes()
        pin, run, artifact = records(raw)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            archive, output = root / "archive.zip", root / "input.age"
            archive.write_bytes(raw)
            with self.assertRaises(ValueError):
                a.unpack_input(pin, dict(run, conclusion="failure"), artifact, archive, output)
            self.assertFalse(output.exists())
            archive.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
            with self.assertRaises(ValueError):
                a.unpack_input(pin, run, artifact, archive, output)
            self.assertFalse(output.exists())

    def test_extra_duplicate_unsafe_and_non_ciphertext_members_reject(self):
        entries = [[("input.age", PAYLOAD), ("extra.json", b"{}")] ,
                   [("input.age", PAYLOAD), ("input.age", PAYLOAD)],
                   [("../input.age", PAYLOAD)], [("nested/input.age", PAYLOAD)],
                   [("input.age/", PAYLOAD)], [("claim.json", b"{}"), ("result.json", b"{}"), ("worker.log", b"failure")]]
        for names in entries:
            with self.subTest(names=[item[0] for item in names]), tempfile.TemporaryDirectory() as temporary:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    raw = zip_bytes(names)
                pin, run, artifact = records(raw)
                root = Path(temporary).resolve()
                archive, output = root / "archive.zip", root / "input.age"
                archive.write_bytes(raw)
                with self.assertRaises(ValueError):
                    a.unpack_input(pin, run, artifact, archive, output)
                self.assertFalse(output.exists())

    def test_special_files_and_unsupported_compression_reject_before_output(self):
        for defect in ("symlink", "fifo", "bzip2", "size"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                entry = zipfile.ZipInfo("input.age")
                if defect in ("symlink", "fifo"):
                    entry.external_attr = ((stat.S_IFLNK if defect == "symlink" else stat.S_IFIFO) | 0o600) << 16
                entry.compress_type = zipfile.ZIP_BZIP2 if defect == "bzip2" else zipfile.ZIP_STORED
                raw = zip_bytes([(entry, PAYLOAD[:-1] if defect == "size" else PAYLOAD)])
                pin, run, artifact = records(raw)
                root = Path(temporary).resolve()
                archive, output = root / "archive.zip", root / "input.age"
                archive.write_bytes(raw)
                with self.assertRaises(ValueError):
                    a.unpack_input(pin, run, artifact, archive, output)
                self.assertFalse(output.exists())

    def test_directory_count_size_zip64_comment_and_trailing_data_fail_before_parser(self):
        raw = zip_bytes()
        cases = [b"short", raw + b"trailing", zip_bytes(comment=b"comment")]
        for offset, fmt, value in ((8, "<H", 65535), (10, "<H", 65535),
                                   (12, "<I", a.MAX_CENTRAL_DIRECTORY + 1),
                                   (16, "<I", 0xFFFFFFFF), (4, "<H", 1)):
            changed = bytearray(raw)
            struct.pack_into(fmt, changed, len(raw) - 22 + offset, value)
            cases.append(bytes(changed))
        for changed in cases:
            with self.subTest(sha=hashlib.sha256(changed).hexdigest()), tempfile.TemporaryDirectory() as temporary:
                pin, run, artifact = records(changed)
                root = Path(temporary).resolve()
                archive, output = root / "archive.zip", root / "input.age"
                archive.write_bytes(changed)
                with patch.object(a.zipfile, "ZipFile", side_effect=AssertionError("unbounded parser reached")), self.assertRaises(ValueError):
                    a.unpack_input(pin, run, artifact, archive, output)
                self.assertFalse(output.exists())

    def test_wrong_inner_hash_keeps_first_output_and_never_verifies(self):
        raw = zip_bytes()
        pin, run, artifact = records(raw)
        pin["ciphertext"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            archive, output = root / "archive.zip", root / "input.age"
            archive.write_bytes(raw)
            with self.assertRaises(ValueError):
                a.unpack_input(pin, run, artifact, archive, output)
            self.assertEqual(output.read_bytes(), PAYLOAD)
            with self.assertRaises(FileExistsError):
                a.unpack_input(pin, run, artifact, archive, output)

    def test_zip64_locator_cannot_override_bounded_directory_before_parse(self):
        raw = zip64_override(zip_bytes([(f"entry-{index:04d}.age", b"") for index in range(1300)]))
        pin, run, artifact = records(raw)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            archive, output = root / "archive.zip", root / "input.age"
            archive.write_bytes(raw)
            with patch.object(a.zipfile, "ZipFile", side_effect=AssertionError("unbounded parser reached")), self.assertRaises(ValueError):
                a.unpack_input(pin, run, artifact, archive, output)
            self.assertFalse(output.exists())

    def test_small_zip64_is_rejected_even_when_both_hashes_match(self):
        raw = zip64_override(zip_bytes())
        pin, run, artifact = records(raw)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            archive, output = root / "archive.zip", root / "input.age"
            archive.write_bytes(raw)
            with self.assertRaises(ValueError):
                a.unpack_input(pin, run, artifact, archive, output)
            self.assertFalse(output.exists())

    def test_changed_archive_during_extraction_never_returns_a_verified_receipt(self):
        raw = zip_bytes()
        pin, run, artifact = records(raw)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            archive, output = root / "archive.zip", root / "input.age"
            archive.write_bytes(raw)
            real_copy = a.t.copy_verified
            def copy_and_change(stream, destination, expected):
                receipt = real_copy(stream, destination, expected)
                archive.write_bytes(raw + b"changed")
                return receipt
            with patch.object(a.t, "copy_verified", side_effect=copy_and_change), self.assertRaises(ValueError):
                a.unpack_input(pin, run, artifact, archive, output)
            self.assertEqual(output.read_bytes(), PAYLOAD)

    def test_links_in_archive_output_or_parents_never_follow(self):
        raw = zip_bytes()
        pin, run, artifact = records(raw)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            archive = root / "archive.zip"
            archive.write_bytes(raw)
            linked_archive = root / "linked.zip"
            linked_archive.symlink_to(archive)
            with self.assertRaises(ValueError):
                a.unpack_input(pin, run, artifact, linked_archive, root / "out.age")
            linked_parent = root / "linked-parent"
            linked_parent.symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                a.unpack_input(pin, run, artifact, archive, linked_parent / "out.age")
            dangling = root / "dangling.age"
            dangling.symlink_to(root / "missing")
            with self.assertRaises(FileExistsError):
                a.unpack_input(pin, run, artifact, archive, dangling)
            self.assertFalse((root / "missing").exists())


if __name__ == "__main__":
    import unittest
    unittest.main()
