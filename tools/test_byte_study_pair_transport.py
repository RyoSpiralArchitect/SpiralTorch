"""Synthetic transport fixtures only; no native imports, device or network."""
import gzip
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
import zipfile

SPEC = importlib.util.spec_from_file_location("pair_transport", Path(__file__).with_name("byte_study_pair_transport.py"))
T = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(T)
ROOT = Path(__file__).resolve().parents[1]


def archive_bytes(rows):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, raw, kind in rows:
            member = tarfile.TarInfo(name)
            member.type, member.mode = kind, 0o600
            member.size = len(raw)
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                member.linkname = "../outside"
            archive.addfile(member, io.BytesIO(raw))
    return gzip.compress(output.getvalue(), mtime=0)


class Transport(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.failures = self.root / "failure"
        (self.failures / "version-9").mkdir(parents=True)
        self.original = {name: (name + "\n").encode() for name in T.ORIGINAL}
        for name, raw in self.original.items():
            (self.failures / "version-9" / name).write_bytes(raw)
        self.wheel = self.root / Path(T.WHEEL).name
        with zipfile.ZipFile(self.wheel, "w") as wheel:
            wheel.writestr("spiraltorch/spiraltorch.abi3.so", b"synthetic-native-not-loadable")
            wheel.writestr("spiraltorch/__init__.py", b"# synthetic only\n")
        for owner, name, value in (
            (T, "ORIGINAL", {name: T.B.digest(raw) for name, raw in self.original.items()}),
            (T.prelude, "WHEEL_SHA256", T.B.digest(self.wheel.read_bytes())),
            (T.prelude, "NATIVE_SHA256", T.B.digest(b"synthetic-native-not-loadable")),
        ):
            patch = mock.patch.object(owner, name, value)
            patch.start()
            self.addCleanup(patch.stop)

    def input_rows(self):
        return [("original/version-9/" + name, raw, tarfile.REGTYPE)
                for name, raw in sorted(self.original.items())] + [(T.WHEEL, self.wheel.read_bytes(), tarfile.REGTYPE)]

    def unpack_rows(self, rows, kind="input", suffix=b""):
        packet = self.root / "packet.gz"
        packet.write_bytes(archive_bytes(rows) + suffix)
        return T.unpack(packet, self.root / "unpacked", T.B.digest(packet.read_bytes()), kind)

    def result_root(self):
        root = self.root / "evidence"
        (root / "diagnostic").mkdir(parents=True)
        (root / "diagnostic" / "result.json").write_bytes(b'{"outcome":"stopped","release_ready":false}\n')
        (root / "failed.log").write_bytes(b"partial failure\n")
        return root

    def result_rows(self, data=None):
        data = data or {"failed.log": b"partial failure\n"}
        manifest = {"schema": T.SCHEMA, "kind": "evidence_only", "release_ready": False,
                    "files": {name: T.identity(raw) for name, raw in data.items()}}
        return [(T.MANIFEST, T.encoded(manifest), tarfile.REGTYPE)] + [
            (name, raw, tarfile.REGTYPE) for name, raw in data.items()]

    def test_exact_nine_file_input_roundtrip(self):
        packet = self.root / "input.gz"
        receipt = T.pack_input(self.failures, self.wheel, packet)
        output = self.root / "decoded"
        result = T.unpack(packet, output, receipt["archive_sha256"], "input")
        self.assertEqual(result["file_count"], 9)
        self.assertFalse(result["release_ready"])
        self.assertTrue(result["transport_verified"])
        self.assertEqual((output / T.WHEEL).read_bytes(), self.wheel.read_bytes())
        self.assertEqual(output.stat().st_mode & 0o777, 0o700)
        self.assertEqual(packet.stat().st_mode & 0o777, 0o600)
        for name, raw in self.original.items():
            path = output / "original/version-9" / name
            self.assertEqual(path.read_bytes(), raw)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_packing_is_deterministic(self):
        first, second = self.root / "first.gz", self.root / "second.gz"
        T.pack_input(self.failures, self.wheel, first)
        os.utime(self.wheel, (500, 500))
        T.pack_input(self.failures, self.wheel, second)
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_wrong_original_hash_is_rejected(self):
        (self.failures / "version-9/request.json").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "original input identity"):
            T.pack_input(self.failures, self.wheel, self.root / "packet.gz")

    def test_wheel_and_native_pins_both_required(self):
        for attribute in ("WHEEL_SHA256", "NATIVE_SHA256"):
            with self.subTest(attribute=attribute), mock.patch.object(T.prelude, attribute, "0" * 64):
                with self.assertRaises(ValueError):
                    T.pack_input(self.failures, self.wheel, self.root / "packet.gz")

    def test_no_output_overwrite(self):
        output = self.root / "output.gz"
        output.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            T.pack_input(self.failures, self.wheel, output)
        self.assertEqual(output.read_bytes(), b"keep")

    def test_input_archive_cannot_modify_original_directory(self):
        with self.assertRaisesRegex(ValueError, "outside the preserved failure"):
            T.pack_input(self.failures, self.wheel, self.failures / "packet.gz")

    def test_source_link_is_rejected(self):
        link = self.root / "wheel-link"
        link.symlink_to(self.wheel)
        with self.assertRaises(ValueError):
            T.pack_input(self.failures, link, self.root / "packet.gz")

    def test_parent_link_is_rejected(self):
        alias = self.root / "alias"
        alias.symlink_to(self.failures, target_is_directory=True)
        with self.assertRaises(ValueError):
            T.pack_input(alias, self.wheel, self.root / "packet.gz")

    def test_input_extra_and_missing_are_rejected(self):
        for rows in (self.input_rows()[:-1], self.input_rows() + [("extra", b"x", tarfile.REGTYPE)]):
            with self.subTest(count=len(rows)), tempfile.TemporaryDirectory(dir=self.root) as temp:
                packet = Path(temp) / "input.gz"
                packet.write_bytes(archive_bytes(rows))
                with self.assertRaises(ValueError):
                    T.unpack(packet, Path(temp) / "decoded", T.B.digest(packet.read_bytes()), "input")

    def test_duplicate_is_rejected(self):
        rows = self.input_rows()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.unpack_rows(rows + [rows[0]])

    def test_traversal_and_ambiguous_names_are_rejected(self):
        for name in ("../outside", "/outside", "a/../b", "a//b", "./x", "a\\b", ".", "a/."):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    T.canonical(name)

    def test_tar_types_are_rejected_before_extraction(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE, tarfile.FIFOTYPE,
                     tarfile.CHRTYPE, tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.GNUTYPE_LONGNAME):
            with self.subTest(kind=kind):
                with gzip.GzipFile(fileobj=io.BytesIO(archive_bytes([("x", b"", kind)]))) as stream:
                    with self.assertRaises(ValueError):
                        list(T.members(stream, 4096, 4096, 2))

    def test_archive_sha_is_checked_before_output_creation(self):
        packet = self.root / "packet.gz"
        packet.write_bytes(archive_bytes(self.input_rows()))
        with self.assertRaisesRegex(ValueError, "archive identity"):
            T.unpack(packet, self.root / "decoded", "0" * 64, "input")
        self.assertFalse((self.root / "decoded").exists())

    def test_changed_member_is_not_accepted(self):
        rows = self.input_rows()
        rows[0] = (rows[0][0], b"changed", tarfile.REGTYPE)
        with self.assertRaisesRegex(ValueError, "member identity"):
            self.unpack_rows(rows)

    def test_compressed_bound(self):
        packet = self.root / "oversized"
        packet.write_bytes(b"123456")
        with self.assertRaises(ValueError):
            T.digest_archive(packet, 5)

    def test_expansion_and_file_count_bounds(self):
        raw = archive_bytes([("x", b"a" * 2048, tarfile.REGTYPE), ("y", b"b", tarfile.REGTYPE)])
        for total, single, count in ((100, 4096, 2), (4096, 100, 2), (4096, 4096, 1)):
            with self.subTest(limits=(total, single, count)), gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
                with self.assertRaises(ValueError):
                    list(T.members(stream, total, single, count))

    def test_extended_header_not_expanded(self):
        info = tarfile.TarInfo("extension")
        info.type, info.size = tarfile.XHDTYPE, 1024 * 1024 * 1024
        stream = io.BytesIO(info.tobuf(format=tarfile.USTAR_FORMAT))
        with self.assertRaises(ValueError):
            list(T.members(stream, T.MAX_RESULT, T.MAX_FILE, 10))
        self.assertEqual(stream.tell(), 512)

    def test_oversized_manifest_rejected_before_reading_payload(self):
        info = tarfile.TarInfo(T.MANIFEST)
        info.size = T.MAX_MANIFEST + 1
        stream = io.BytesIO(info.tobuf(format=tarfile.USTAR_FORMAT))
        with self.assertRaises(ValueError):
            list(T.members(stream, T.MAX_RESULT, T.MAX_FILE, 10))
        self.assertEqual(stream.tell(), 512)

    def test_gzip_checksum_corruption_rejected(self):
        raw = bytearray(archive_bytes(self.input_rows()))
        raw[-8] ^= 1
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            with self.assertRaises(gzip.BadGzipFile):
                list(T.members(stream, T.MAX_INPUT, T.MAX_FILE, 10))

    def test_truncated_archive_rejected(self):
        raw = gzip.decompress(archive_bytes(self.input_rows()))
        for size in (100, 600, len(raw.rstrip(b"\0"))):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    list(T.members(io.BytesIO(raw[:size]), T.MAX_INPUT, T.MAX_FILE, 10))

    def test_hidden_trailing_archive_and_zero_bomb_rejected(self):
        raw = gzip.decompress(archive_bytes(self.input_rows()))
        for suffix in (b"hidden", bytes(20000)):
            with self.subTest(length=len(suffix)):
                with self.assertRaises(ValueError):
                    list(T.members(io.BytesIO(raw + suffix), T.MAX_INPUT, T.MAX_FILE, 10))

    def test_result_roundtrip_preserves_partial_failure(self):
        root, archive = self.result_root(), self.root / "result.gz"
        receipt = T.pack_result(root, archive)
        result = T.unpack(archive, self.root / "decoded", receipt["archive_sha256"], "result")
        self.assertFalse(result["release_ready"])
        self.assertEqual(result["file_count"], 2)
        for path in root.rglob("*"):
            if path.is_file():
                self.assertEqual(path.read_bytes(), (self.root / "decoded" / path.relative_to(root)).read_bytes())

    def test_result_output_cannot_be_inside_source(self):
        root = self.result_root()
        with self.assertRaises(ValueError):
            T.pack_result(root, root / "result.gz")

    def test_result_refuses_links_special_files_and_reserved_manifest(self):
        root = self.result_root()
        path = root / "bad"
        path.symlink_to(self.wheel)
        with self.assertRaises(ValueError):
            T.pack_result(root, self.root / "result.gz")
        path.unlink()
        os.mkfifo(path)
        with self.assertRaises(ValueError):
            T.pack_result(root, self.root / "result.gz")
        path.unlink()
        (root / T.MANIFEST).write_bytes(b"{}")
        with self.assertRaises(ValueError):
            T.pack_result(root, self.root / "result.gz")

    def test_source_content_change_aborts_pack(self):
        root = self.result_root()
        original = T.raw_file
        counts = {}

        def change(path, *args, **kwargs):
            counts[path] = counts.get(path, 0) + 1
            if path.name == "failed.log" and counts[path] == 2:
                path.write_bytes(b"changed")
            return original(path, *args, **kwargs)

        with mock.patch.object(T, "raw_file", side_effect=change):
            with self.assertRaisesRegex(ValueError, "file changed"):
                T.pack_result(root, self.root / "result.gz")

    def test_source_roster_change_aborts_pack(self):
        root = self.result_root()
        original = T.write_archive

        def change(*args, **kwargs):
            result = original(*args, **kwargs)
            (root / "new.log").write_bytes(b"changed")
            return result

        with mock.patch.object(T, "write_archive", side_effect=change):
            with self.assertRaisesRegex(ValueError, "roster changed"):
                T.pack_result(root, self.root / "result.gz")

    def test_result_requires_first_manifest(self):
        rows = self.result_rows()
        with self.assertRaisesRegex(ValueError, "manifest must be first"):
            self.unpack_rows(rows[1:] + rows[:1], "result")

    def test_result_missing_member(self):
        with self.assertRaisesRegex(ValueError, "missing archive member"):
            self.unpack_rows(self.result_rows()[:1], "result")

    def test_result_extra_member(self):
        with self.assertRaisesRegex(ValueError, "unexpected archive member"):
            self.unpack_rows(self.result_rows() + [("unlisted", b"extra", tarfile.REGTYPE)], "result")

    def test_result_size_and_hash_recomputed(self):
        rows = self.result_rows()
        rows[1] = (rows[1][0], b"different", tarfile.REGTYPE)
        with self.assertRaisesRegex(ValueError, "member identity"):
            self.unpack_rows(rows, "result")

    def test_manifest_duplicate_json_and_invalid_claims_rejected(self):
        raw = self.result_rows()[0][1]
        mutations = [raw.replace(b'"release_ready": false', b'"release_ready": false, "release_ready": false')]
        value = json.loads(raw)
        for field, change in (("release_ready", True), ("kind", "qualified"), ("schema", "unknown")):
            mutations.append(T.encoded(dict(value, **{field: change})))
        for mutation in mutations:
            with self.subTest(raw=mutation):
                with self.assertRaises(ValueError):
                    T.expected_result(mutation)

    def test_manifest_rejects_boolean_size_nonfinite_and_bad_hash(self):
        for row in ({"bytes": True, "sha256": "a" * 64}, {"bytes": 2, "sha256": "invalid"},
                    {"bytes": -1, "sha256": "a" * 64}, {"bytes": T.MAX_FILE + 1, "sha256": "a" * 64}):
            value = {"schema": T.SCHEMA, "kind": "evidence_only", "release_ready": False, "files": {"x": row}}
            with self.subTest(row=row), self.assertRaises(ValueError):
                T.expected_result(T.encoded(value))

    def test_existing_extraction_directory_not_reused(self):
        packet = self.root / "packet.gz"
        packet.write_bytes(archive_bytes(self.input_rows()))
        output = self.root / "decoded"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            T.unpack(packet, output, T.B.digest(packet.read_bytes()), "input")

    def test_archive_symlink_rejected(self):
        source = self.root / "source.gz"
        source.write_bytes(archive_bytes(self.input_rows()))
        link = self.root / "link.gz"
        link.symlink_to(source)
        with self.assertRaises(ValueError):
            T.unpack(link, self.root / "decoded", T.B.digest(source.read_bytes()), "input")


class Template(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "docs/ops/byte-study-prelude-pair.workflow.yml"
        cls.text = cls.path.read_text()

    def test_template_is_inert_and_still_has_placeholders(self):
        self.assertNotIn(".github/workflows", str(self.path))
        for placeholder in ("__INPUT_CIPHERTEXT_SHA256__", "__INPUT_PLAINTEXT_SHA256__", "__RESULT_RECIPIENT__"):
            self.assertIn(placeholder, self.text)
        self.assertIn('[[ "$INPUT_PLAINTEXT_SHA256" =~ ^[0-9a-f]{64}$ ]]', self.text)

    def test_one_pair_only_with_deadlines_and_admission(self):
        self.assertEqual(self.text.count("--execute-pair"), 1)
        self.assertIn("--timeout-seconds 5400", self.text)
        self.assertIn("timeout-minutes: 210", self.text)
        self.assertIn('test "$GITHUB_RUN_ATTEMPT" = "1"', self.text)
        self.assertIn('test "$EXPECTED_HARNESS" = "$GITHUB_SHA"', self.text)
        self.assertIn('test "$GITHUB_REF" = "refs/heads/spiralreality/retained-prelude-pair-20261024"', self.text)
        self.assertNotRegex(self.text, r"(?m)^  (push|pull_request|schedule):")

    def test_no_build_no_publishing_no_dependency_fetch(self):
        for forbidden in ("cargo ", "rust-toolchain", "maturin", "twine", "id-token:", "contents: write", "publish_pypi"):
            self.assertNotIn(forbidden, self.text)
        self.assertIn("pip install --no-index --no-deps --no-cache-dir", self.text)
        self.assertIn("python-version: \"3.12.10\"", self.text)
        self.assertIn("ref: " + T.prelude.retained.SOURCE_COMMIT, self.text)

    def test_keys_separated_and_only_ciphertext_uploaded(self):
        self.assertIn('input_key="$TRANSPORT_ROOT/input-key.txt"', self.text)
        self.assertIn('trap \'rm -f "$input_key"\' EXIT', self.text)
        self.assertIn("unset INPUT_IDENTITY", self.text)
        self.assertNotIn("RESULT_IDENTITY", self.text)
        upload = self.text.split("- name: Upload ciphertext only", 1)[1]
        self.assertEqual(re.findall(r"(?m)^            \$\{\{ runner.temp \}\}/(.+)$", upload), [
            "prelude-pair-20261024/public/evidence.tar.gz.age", "prelude-pair-20261024/public/SHA256SUMS"])
        self.assertIn("always()", upload)

    def test_recipient_is_validated_cryptographically_before_native_work(self):
        before = self.text.split("- name: Validate result recipient before any diagnosis", 1)[1]
        probe = before.split("      - uses:", 1)[0]
        self.assertIn('"$TRANSPORT_ROOT/age/age" --encrypt --recipient "$RESULT_RECIPIENT"', probe)
        self.assertIn('"$TRANSPORT_ROOT/result-recipient-probe.age"', probe)
        self.assertIn("- name: Execute the ordinary and retained arms once each", before)
        self.assertNotIn("continue-on-error", self.text)

    def test_shell_blocks_parse_without_executing(self):
        blocks = re.findall(r"(?m)^        run: \|\n((?:          .*\n)+)", self.text)
        self.assertEqual(len(blocks), 8)
        for block in blocks:
            parsed = subprocess.run(["bash", "-n"], input="".join(line[10:] + "\n" for line in block.splitlines()),
                                    text=True, capture_output=True, check=False)
            self.assertEqual(parsed.returncode, 0, parsed.stderr)

    def test_transport_imported_no_native_or_foreign_ml(self):
        self.assertFalse(any(name.split(".")[0] in {"spiraltorch", "torch", "numpy", "tensorflow", "jax"}
                             for name in sys.modules))


if __name__ == "__main__":
    unittest.main()
