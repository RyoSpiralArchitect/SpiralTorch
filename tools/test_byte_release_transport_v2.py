"""CPU-only v2 transport tests. All archives, approvals and repositories are synthetic."""
import copy
import hashlib
import io
from pathlib import Path
import socket
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import byte_release_transport_v2 as t
import byte_release_transport_workflow_v2 as workflow
from test_byte_release_contract import fabricated_admission
from test_byte_release_handoff import accepted_ledger

c = t.c
ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FILES = (
    "tools/byte_release_transport_v2.py", "tools/byte_release_hosted_v2.py",
    "tools/byte_release_transport_workflow_v2.py",
)


def records(admission=None, archive=None, local_sha="f" * 64):
    admission = admission or fabricated_admission()
    descriptor = dict(schema=t.DESCRIPTOR_SCHEMA, candidate=admission["candidate"],
                      admission_sha256=c.digest(admission), source_commit=admission["source"]["commit"],
                      plaintext=t.v1.regular(archive, t.MAX_INPUT) if archive else {"bytes": 1024, "sha256": "d" * 64},
                      ciphertext={"bytes": 8, "sha256": hashlib.sha256(b"testonly").hexdigest()},
                      object={"host": "storage.example.test", "path": "/objects/synthetic.age"}, release_ready=False)
    amendment = dict(schema=t.AMENDMENT_SCHEMA, decision=c.DECISION, candidate=admission["candidate"],
                     admission_sha256=c.digest(admission), local_evidence_sha256=local_sha,
                     source_commit=admission["source"]["commit"], source_tree=admission["source"]["tree"],
                     controller_commit="e" * 40, controller_tree="f" * 40, descriptor_sha256=c.digest(descriptor),
                     review_record_sha256="1" * 64, result_recipient="age1" + "q" * 58,
                     repository=t.REPO, workflow=t.WORKFLOW,
                     branch=t.branch(admission["candidate"]), concurrency=t.CONCURRENCY,
                     limits=dict(t.LIMITS), release_ready=False)
    return amendment, descriptor


def git(root, *args):
    return subprocess.check_output(
        ["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false",
         "-c", "user.name=Transport CPU Test", "-c", "user.email=synthetic@example.invalid", *args],
        cwd=root, stderr=subprocess.PIPE)


def commit(root):
    git(root, "add", ".")
    git(root, "commit", "-qm", "synthetic transport fixture")
    return t.clean_tree(root)


def controller_fixture(root):
    source = root / "source"
    source.mkdir()
    git(source, "init", "-q")
    path = source / ".github/workflows" / t.WORKFLOW
    path.parent.mkdir(parents=True)
    path.write_bytes((ROOT / ".github/workflows" / t.WORKFLOW).read_bytes())
    (source / "Cargo.lock").write_text("synthetic only\n")
    source_head, source_tree = commit(source)
    controller = root / "controller"
    git(root, "clone", "--quiet", "--no-hardlinks", str(source), str(controller))
    amendment, descriptor = records()
    descriptor["source_commit"] = source_head
    amendment.update(source_commit=source_head, source_tree=source_tree, descriptor_sha256=c.digest(descriptor))
    c.save(controller / t.DESCRIPTOR, descriptor)
    (controller / ".github/workflows" / t.WORKFLOW).write_text(workflow.render(path.read_text()))
    for name in RUNTIME_FILES:
        destination = controller / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    head, tree = commit(controller)
    amendment.update(controller_commit=head, controller_tree=tree)
    return source, controller, amendment, descriptor


class Offline(unittest.TestCase):
    def setUp(self):
        self.network = patch.object(socket, "create_connection", side_effect=AssertionError("network forbidden"))
        self.network.start()
        self.addCleanup(self.network.stop)
        self.native = patch.dict(sys.modules, {name: None for name in
                    ("spiraltorch", "torch", "numpy", "transformers", "peft", "accelerate", "safetensors")})
        self.native.start()
        self.addCleanup(self.native.stop)


class DescriptorOffline(Offline):
    def test_exact_policy_and_external_pin(self):
        amendment, descriptor = records()
        self.assertIs(t.validate_amendment(amendment, c.digest(amendment), descriptor), amendment)
        with self.assertRaisesRegex(ValueError, "external pin"):
            t.validate_amendment(amendment, "0" * 64, descriptor)
        changes = [("candidate", True), ("repository", "other/repo"), ("workflow", "v2.yml"),
                   ("branch", "fresh-retry"), ("concurrency", "fresh-group"),
                   ("release_ready", True), ("source_tree", "bad"), ("review_record_sha256", None),
                   ("source_commit", "0" * 40), ("admission_sha256", "0" * 64),
                   ("descriptor_sha256", "0" * 64), ("unexpected", 1)]
        for key, value in changes:
            changed = copy.deepcopy(amendment)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                t.validate_amendment(changed, c.digest(changed), descriptor)
        for value in (t.MAX_INPUT + 1, float(t.MAX_INPUT)):
            changed = copy.deepcopy(amendment)
            changed["limits"]["input_bytes"] = value
            with self.assertRaises(ValueError):
                t.validate_amendment(changed, c.digest(changed), descriptor)

    def test_descriptor_types_endpoints_and_capacity(self):
        _amendment, descriptor = records()
        for value in (True, 0, -1, 1.5, t.MAX_INPUT + 1):
            changed = copy.deepcopy(descriptor)
            changed["plaintext"]["bytes"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                t.validate_descriptor(changed)
        descriptor["plaintext"]["bytes"] = t.MAX_INPUT
        t.validate_descriptor(descriptor)
        for location in ({"host": "127.0.0.1", "path": "/x.age"},
                         {"host": "storage.example.test:443", "path": "/x.age"},
                         {"host": "storage..test", "path": "/x.age"},
                         {"host": "storage.example.test", "path": "/../x.age"},
                         {"host": "storage.example.test", "path": "/x.age?key=secret"}):
            changed = copy.deepcopy(descriptor)
            changed["object"] = location
            with self.subTest(location=location), self.assertRaises(ValueError):
                t.validate_descriptor(changed)

    def test_private_query_is_not_descriptor_identity(self):
        _amendment, descriptor = records()
        url = "https://storage.example.test/objects/synthetic.age?credential=test-only"
        self.assertEqual(t.validate_url(url, descriptor), url)
        for other in (url.replace("https:", "http:"), url.replace("storage.", "other."),
                      url.replace("/objects/", "/different/"), url + "#fragment",
                      url.replace("https://", "https://username:password@"), "\n" + url):
            with self.subTest(url=other), self.assertRaises(ValueError):
                t.validate_url(other, descriptor)
        self.assertNotIn("credential", c.encoded(descriptor).decode())

    def test_legacy_numeric_addresses_are_not_approved_dns_hosts(self):
        _amendment, descriptor = records()
        for host in ("127.1", "127.000.000.001", "0x7f.0.0.1", "169.254.43518",
                     "0177.0.0.1", "0x7f000001", "2130706433"):
            changed = copy.deepcopy(descriptor)
            changed["object"]["host"] = host
            with self.subTest(host=host), self.assertRaises(ValueError):
                t.validate_descriptor(changed)

    def test_json_duplicate_and_nonfinite_records_reject(self):
        for raw in ('{"a":1,"a":2}', '{"a":NaN}', "x" * (t.MAX_RECORD + 1),
                    "\N{GREEK SMALL LETTER PI}" * (t.MAX_RECORD // 2 + 1)):
            with self.assertRaises(ValueError):
                t.parse_record(raw)

    def test_stream_eof_hash_overflow_and_no_overwrite(self):
        _amendment, descriptor = records()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output = root / "cipher.age"
            self.assertEqual(t.copy_verified(io.BytesIO(b"testonly"), output, descriptor["ciphertext"]),
                             descriptor["ciphertext"])
            with self.assertRaises(FileExistsError):
                t.copy_verified(io.BytesIO(b"testonly"), output, descriptor["ciphertext"])
            for index, data in enumerate((b"test", b"testonly!", b"TESTonly")):
                target = root / str(index)
                with self.assertRaises(ValueError):
                    t.copy_verified(io.BytesIO(data), target, descriptor["ciphertext"])
                self.assertTrue(target.exists())
            link = root / "link"
            link.symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                t.copy_verified(io.BytesIO(b"testonly"), link / "forbidden.age", descriptor["ciphertext"])


class ArchiveOffline(Offline):
    def test_complete_input_roundtrip_and_pinned_local_chain(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            ledger, wheel, admission = accepted_ledger(root)
            packet = root / "input.zip"
            t.v1.pack_input(ledger, wheel, packet)
            local = c.read(ledger / "candidate-1/local-metal.result.json")
            amendment, descriptor = records(admission, packet, local["evidence_sha256"])
            output = root / "decoded"
            t.unpack_input(packet, output, descriptor)
            self.assertEqual(t.verify_input_tree(output, amendment, descriptor), admission)
            self.assertFalse((ledger / "candidate-1/hosted-metal.start.json").exists())
            with self.assertRaises(FileExistsError):
                t.unpack_input(packet, output, descriptor)
            changed = copy.deepcopy(amendment)
            changed["local_evidence_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "complete local proof"):
                t.verify_input_tree(output, changed, descriptor)
            retained = next((output / "ledger/candidate-1/local-metal").rglob("worker-result.json"))
            retained.write_bytes(retained.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                t.verify_input_tree(output, amendment, descriptor)

    def test_real_over_v1_size_fixture_needs_no_limit_monkeypatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            chunk, count = bytes(1024 * 1024), 97
            digest = hashlib.sha256()
            for _ in range(count):
                digest.update(chunk)
            _amendment, descriptor = records()
            manifest = dict(schema=t.v1.SCHEMA, kind="input", admission_sha256=descriptor["admission_sha256"],
                            release_ready=False, files={"data.bin": {"bytes": len(chunk) * count,
                                                                    "sha256": digest.hexdigest()}})
            packet = root / "over-v1.zip"
            with zipfile.ZipFile(packet, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr(t.v1.MANIFEST, c.encoded(manifest))
                with archive.open("data.bin", "w") as member:
                    for _ in range(count):
                        member.write(chunk)
            descriptor["plaintext"] = t.v1.regular(packet, t.MAX_INPUT)
            self.assertGreater(packet.stat().st_size, t.v1.MAX_INPUT_ARCHIVE)
            with self.assertRaises(ValueError):
                t.v1.unpack(packet, root / "v1", descriptor["plaintext"]["sha256"],
                             "input", descriptor["admission_sha256"])
            t.unpack_input(packet, root / "v2", descriptor)
            self.assertEqual(t.v1.regular(root / "v2/data.bin"), manifest["files"]["data.bin"])
            self.assertEqual(t.v1.MAX_INPUT_ARCHIVE, 96 * 1024 * 1024)

    def test_wrong_outer_identity_stops_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            packet = root / "input.zip"
            packet.write_bytes(b"not-a-zip")
            _amendment, descriptor = records()
            with self.assertRaises(ValueError):
                t.unpack_input(packet, root / "out", descriptor)
            self.assertFalse((root / "out").exists())
            with packet.open("wb") as stream:
                stream.truncate(t.MAX_INPUT + 1)
            with self.assertRaises(ValueError):
                t.unpack_input(packet, root / "out", descriptor)

    def test_unsafe_and_incomplete_archives_reject(self):
        defects = ("traversal", "link", "special", "duplicate", "missing", "extra", "private_key",
                   "hash", "size", "kind", "admission", "compression", "manifest_duplicate")
        for defect in defects:
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                _amendment, descriptor = records()
                name = "../escape" if defect == "traversal" else "input-key.txt" if defect == "private_key" else "data.json"
                files = {name: {"bytes": 3 if defect == "size" else 2, "sha256": hashlib.sha256(b"{}").hexdigest()}}
                manifest = dict(schema=t.v1.SCHEMA, kind="result" if defect == "kind" else "input",
                                admission_sha256="0" * 64 if defect == "admission" else descriptor["admission_sha256"],
                                release_ready=False, files=files)
                packet = root / "bad.zip"
                with zipfile.ZipFile(packet, "w") as archive, warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    encoded = c.encoded(manifest)
                    if defect == "manifest_duplicate":
                        encoded = b'{"kind":"input",' + encoded[1:]
                    archive.writestr(t.v1.MANIFEST, encoded)
                    if defect != "missing":
                        info = zipfile.ZipInfo(name)
                        if defect in ("link", "special"):
                            info.create_system = 3
                            info.external_attr = ((stat.S_IFLNK if defect == "link" else stat.S_IFIFO) | 0o600) << 16
                        if defect == "compression":
                            info.compress_type = zipfile.ZIP_BZIP2
                        archive.writestr(info, b"[]" if defect == "hash" else b"{}")
                    if defect == "duplicate":
                        archive.writestr(name, b"{}")
                    if defect == "extra":
                        archive.writestr("extra.json", b"{}")
                descriptor["plaintext"] = t.v1.regular(packet, t.MAX_INPUT)
                with self.assertRaises(ValueError):
                    t.unpack_input(packet, root / "out", descriptor)

    def test_declared_expansion_and_member_count_stop_before_member_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            packet = root / "input.zip"
            packet.write_bytes(b"synthetic metadata-only boundary")
            _amendment, descriptor = records()
            descriptor["plaintext"] = t.v1.regular(packet, t.MAX_INPUT)
            for defect in ("member", "total", "manifest", "count"):
                manifest = zipfile.ZipInfo(t.v1.MANIFEST)
                manifest.file_size = t.LIMITS["manifest_bytes"] + 1 if defect == "manifest" else 8
                members = [manifest]
                count = t.v1.MAX_FILES + 1 if defect == "count" else 9 if defect == "total" else 1
                for index in range(count):
                    info = zipfile.ZipInfo(f"data-{index}.bin")
                    info.file_size = t.v1.MAX_FILE + 1 if defect == "member" else t.v1.MAX_FILE if defect == "total" else 2
                    members.append(info)
                with self.subTest(defect=defect), patch.object(t.zipfile, "ZipFile") as archive:
                    handle = archive.return_value.__enter__.return_value
                    handle.infolist.return_value = members
                    with self.assertRaises(ValueError):
                        t.unpack_input(packet, root / defect, descriptor)
                    handle.read.assert_not_called()
                    handle.open.assert_not_called()


class ControllerOffline(Offline):
    def test_each_runtime_module_is_required_in_the_committed_controller(self):
        for name in RUNTIME_FILES:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                source, controller, amendment, descriptor = controller_fixture(Path(temporary).resolve())
                (controller / name).unlink()
                head, tree = commit(controller)
                amendment.update(controller_commit=head, controller_tree=tree)
                with self.assertRaisesRegex(ValueError, "runtime modules"):
                    t.validate_controller(controller, source, amendment, descriptor, c.digest(amendment))

    def test_clean_two_root_controller_and_acyclic_pin(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, controller, amendment, descriptor = controller_fixture(Path(temporary).resolve())
            self.assertEqual(t.validate_controller(controller, source, amendment, descriptor, c.digest(amendment)),
                             amendment["controller_commit"])
            self.assertNotIn(amendment["controller_commit"], (controller / t.DESCRIPTOR).read_text())
            self.assertEqual(t.clean_tree(source), [amendment["source_commit"], amendment["source_tree"]])

    def test_wrong_pin_and_uncommitted_source_reject(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, controller, amendment, descriptor = controller_fixture(Path(temporary).resolve())
            with self.assertRaisesRegex(ValueError, "external pin"):
                t.validate_controller(controller, source, amendment, descriptor, "0" * 64)
            (source / "Cargo.lock").write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "dirty"):
                t.validate_controller(controller, source, amendment, descriptor, c.digest(amendment))

    def test_reviewed_controller_cannot_change_execution_envelope(self):
        mutations = (
            lambda text: text.replace("macos-26", "macos-latest"),
            lambda text: text.replace("3.12.10", "3.12.11"),
            lambda text: text.replace("timeout-minutes: 270", "timeout-minutes: 300"),
            lambda text: text.replace("cancel-in-progress: false", "cancel-in-progress: true"),
            lambda text: text.replace("group: byte-release-qualification-20261025", "group: fresh"),
            lambda text: text.replace("--no-index --no-deps", "--no-deps"),
            lambda text: text + "\n# tools/run_byte_release_qualification.py --environment hosted-metal --execute\n",
        )
        with tempfile.TemporaryDirectory() as temporary:
            source, controller, amendment, descriptor = controller_fixture(Path(temporary).resolve())
            path = controller / ".github/workflows" / t.WORKFLOW
            baseline = path.read_text()
            for mutate in mutations:
                path.write_text(mutate(baseline))
                head, tree = commit(controller)
                changed = dict(amendment, controller_commit=head, controller_tree=tree)
                with self.assertRaisesRegex(ValueError, "execution envelope"):
                    t.validate_controller(controller, source, changed, descriptor, c.digest(changed))

    def test_tree_allowlist_rejects_nontransport_change_or_link(self):
        for defect in ("source", "extra", "link"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                source, controller, amendment, descriptor = controller_fixture(Path(temporary).resolve())
                if defect == "source":
                    (controller / "Cargo.lock").write_text("changed\n")
                elif defect == "extra":
                    (controller / "arbitrary.py").write_text("raise RuntimeError()\n")
                else:
                    path = controller / t.DESCRIPTOR
                    original = path.read_bytes()
                    path.unlink()
                    target = Path(temporary).resolve() / "descriptor.json"
                    target.write_bytes(original)
                    path.symlink_to(target)
                head, tree = commit(controller)
                amendment.update(controller_commit=head, controller_tree=tree)
                with self.assertRaises(ValueError):
                    t.validate_controller(controller, source, amendment, descriptor, c.digest(amendment))

    def test_renderer_preserves_execution_step_exactly_and_requires_known_structure(self):
        baseline = (ROOT / ".github/workflows" / t.WORKFLOW).read_text()
        rendered = workflow.render(baseline)
        start = "      - name: Install the same wheel offline without cached Python code\n"
        end = "      - name: Encrypt the first outcome including incomplete evidence\n"
        self.assertEqual(baseline[baseline.index(start):baseline.index(end)],
                         rendered[rendered.index(start):rendered.index(end)])
        self.assertEqual(rendered.count("--environment hosted-metal --execute"), 1)
        self.assertNotIn("      - name: Decrypt the packet", rendered)
        self.assertLess(rendered.index("path: source"), rendered.index("admit-run"))
        self.assertLess(rendered.index("admit-run"), rendered.index("INPUT_OBJECT_URL:"))
        self.assertLess(rendered.index("unset INPUT_OBJECT_URL"), rendered.index("prepare-input"))
        with self.assertRaises(ValueError):
            workflow.render(rendered)


if __name__ == "__main__":
    unittest.main()
