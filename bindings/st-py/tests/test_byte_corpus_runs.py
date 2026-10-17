"""Store faults use a fake owner; GPU resume evidence is a separate qualification."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "byte_run_store_test_subject",
    Path(__file__).resolve().parents[1] / "spiraltorch" / "byte_corpus_runs.py",
)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


class FakeStudy:
    total_updates = 4
    calls = []
    fail_at = None
    corrupt_at = None
    before_advance = None

    @classmethod
    def from_json(cls, raw):
        study = cls()
        study.request_sha256 = runner._digest(raw)
        return study

    def validate_segment(self, stop, *, resume=None):
        if not 0 <= stop <= self.total_updates:
            raise ValueError("out of range")
        if resume is not None and resume.request_sha256 != self.request_sha256:
            raise ValueError("foreign Rust checkpoint")

    def checkpoint_from_json(self, raw):
        value = json.loads(raw)
        if value.get("invalid_geometry"):
            raise ValueError("Rust rejected synthetic invalid geometry")
        return SimpleNamespace(completed_updates=value["cursor"],
                               request_sha256=value["request_sha256"],
                               to_json=lambda: raw.decode())

    def advance_wgpu(self, stop, *, resume):
        if self.before_advance is not None:
            self.before_advance()
        self.calls.append((stop, None if resume is None else resume.completed_updates))
        if self.fail_at == stop:
            raise RuntimeError("simulated GPU failure")
        raw = json.dumps(dict(cursor=stop + (self.corrupt_at == stop),
                              request_sha256=self.request_sha256, signed_zero=-0.0)).encode()
        return SimpleNamespace(checkpoint=self.checkpoint_from_json(raw),
                               report_json=' {"synthetic": true, "signed_zero": -0.0}\n')


@unittest.skipUnless(os.name == "posix", "the durable store currently requires POSIX")
class StoreTests(unittest.TestCase):
    def setUp(self):
        FakeStudy.calls = []
        FakeStudy.fail_at = FakeStudy.corrupt_at = FakeStudy.before_advance = None
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.request = self.root / "request.json"
        self.request.write_bytes(b' {"synthetic": true}\n')
        self.output = self.root / "run"
        self.addCleanup(patch.stopall)
        patch.object(runner, "_study_type", return_value=FakeStudy).start()
        self.quiet = redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    def run_store(self, stops=(0, 2, 4), **kwargs):
        return runner.run_byte_corpus_segments(self.request, self.output, stops, **kwargs)

    def test_claim_before_gpu_private_modes_exact_bytes_and_load(self):
        def claimed(_):
            self.assertTrue((self.output / "claim.json").exists())
            self.assertEqual((self.output / "request.json").read_bytes(), self.request.read_bytes())
            self.assertFalse((self.output / "completed.json").exists())
        FakeStudy.before_advance = claimed
        receipts = self.run_store()
        self.assertEqual(FakeStudy.calls, [(0, None), (2, 0), (4, 2)])
        self.assertEqual([r["stop_after"] for r in receipts], [0, 2, 4])
        self.assertEqual(stat.S_IMODE(self.output.stat().st_mode), 0o700)
        for path in self.output.iterdir():
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertFalse(path.name.startswith(".pending-"))
        saved = runner.load_byte_corpus_segment(self.output, 2)
        self.assertEqual(saved.checkpoint.to_json().encode(),
                         (self.output / "checkpoint-2.json").read_bytes())
        self.assertEqual(saved.report_json.encode(), (self.output / "report-2.json").read_bytes())
        with self.assertRaises(FileExistsError):
            self.run_store()
        self.assertEqual(len(FakeStudy.calls), 3)

    def test_explicit_resume_preserves_source_and_records_parent(self):
        FakeStudy.fail_at = 4
        with self.assertRaisesRegex(RuntimeError, "simulated"):
            self.run_store()
        snapshot = {p.name: p.read_bytes() for p in self.output.iterdir()}
        saved = runner.load_byte_corpus_segment(self.output, 2)
        FakeStudy.fail_at = None
        child = self.root / "child"
        runner.run_byte_corpus_segments(self.request, child, [4], resume_from=self.output,
                                       resume_after=2, expected_resume_sha256=saved.receipt_sha256)
        self.assertEqual(FakeStudy.calls[-1], (4, 2))
        self.assertEqual(snapshot, {p.name: p.read_bytes() for p in self.output.iterdir()})
        parent = json.loads((child / "claim.json").read_bytes())["parent"]
        self.assertEqual(parent, dict(claim_sha256=saved.claim_sha256,
                                     segment_sha256=saved.receipt_sha256,
                                     checkpoint_sha256=saved.checkpoint_sha256,
                                     completed_updates=2))
        self.assertEqual(runner.load_byte_corpus_segment(child, 4).checkpoint.completed_updates, 4)

    def test_invalid_cursors_and_infinite_stops_fail_before_claim(self):
        from itertools import count
        for stops in ([], [2], [4, 4], [4, 2], [-1, 4], [True, 4], count()):
            with self.subTest(stops=stops), self.assertRaises(ValueError):
                self.run_store(stops)
        self.assertFalse(self.output.exists())
        self.assertEqual(FakeStudy.calls, [])

    def test_resume_requires_both_arguments_and_forward_progress(self):
        for options in (dict(resume_after=2), dict(resume_from=self.root),
                        dict(expected_resume_sha256="0" * 64)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_store(**options)
        self.assertFalse(self.output.exists())
        self.run_store()
        for stop, stops in ((True, [4]), (-1, [4]), (2, [2, 4]), (4, [4])):
            child = self.root / "child"
            with self.subTest(stop=stop, stops=stops), self.assertRaises(ValueError):
                runner.run_byte_corpus_segments(self.request, child, stops,
                                               resume_from=self.output, resume_after=stop)
            self.assertFalse(child.exists())

    def test_changed_request_or_wrong_pin_fail_before_claim(self):
        self.run_store()
        child = self.root / "child"
        for pin in ("0" * 64, "SHA256", True):
            with self.subTest(pin=pin), self.assertRaises(ValueError):
                runner.run_byte_corpus_segments(self.request, child, [4],
                                               resume_from=self.output, resume_after=2,
                                               expected_resume_sha256=pin)
        self.request.write_bytes(self.request.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "exact original"):
            runner.run_byte_corpus_segments(self.request, child, [4],
                                           resume_from=self.output, resume_after=2)
        self.assertFalse(child.exists())
        self.assertEqual(len(FakeStudy.calls), 3)

    def test_failed_publication_never_adopts_unsealed_files(self):
        original = runner._write_new
        for name in ("claim.json", "request.json", "report-2.json", "checkpoint-2.json",
                     "segment-2.json", "completed.json"):
            for after in (False, True):
                with self.subTest(name=name, after=after):
                    self.output = self.root / f"fault-{name}-{after}"
                    def fail(directory, target, raw):
                        if target == name and not after:
                            raise OSError("injected before publication")
                        result = original(directory, target, raw)
                        if target == name and after:
                            raise OSError("injected after publication")
                        return result
                    with patch.object(runner, "_write_new", side_effect=fail):
                        with self.assertRaisesRegex(OSError, "injected"):
                            self.run_store()
                    if name in ("claim.json", "request.json"):
                        self.assertFalse((self.output / "segment-0.json").exists())
                    else:
                        self.assertEqual(runner.load_byte_corpus_segment(
                            self.output, 0).checkpoint.completed_updates, 0)
                        sealed = name == "completed.json" or (name == "segment-2.json" and after)
                        if sealed:
                            self.assertEqual(runner.load_byte_corpus_segment(
                                self.output, 2).checkpoint.completed_updates, 2)
                        else:
                            with self.assertRaises(FileNotFoundError):
                                runner.load_byte_corpus_segment(self.output, 2)
                    with self.assertRaises(FileExistsError):
                        self.run_store()

    def test_hash_corruption_and_symlinks_are_rejected(self):
        self.run_store()
        for name in ("request.json", "claim.json", "report-2.json", "checkpoint-2.json",
                     "segment-2.json"):
            path = self.output / name
            original = path.read_bytes()
            path.write_bytes(original + b"corrupt")
            with self.subTest(name=name), self.assertRaises(ValueError):
                runner.load_byte_corpus_segment(self.output, 2)
            path.unlink()
            target = self.root / ("target-" + name)
            target.write_bytes(original)
            path.symlink_to(target)
            with self.assertRaises(OSError):
                runner.load_byte_corpus_segment(self.output, 2)
            path.unlink()
            path.write_bytes(original)
        alias = self.root / "alias"
        alias.symlink_to(self.output, target_is_directory=True)
        with self.assertRaises(OSError):
            runner.load_byte_corpus_segment(alias, 2)

    def test_size_limit_fifo_and_no_replace(self):
        self.run_store()
        with runner._directory(self.output) as directory:
            with self.assertRaises(ValueError):
                runner._read(directory, "checkpoint-2.json", 1)
            with self.assertRaises(FileExistsError):
                runner._write_new(directory, "request.json", b"not a replacement")
            os.mkfifo(self.output / "fifo", mode=0o600)
            with self.assertRaisesRegex(ValueError, "regular file"):
                runner._read(directory, "fifo", 100)
        self.assertEqual((self.output / "request.json").read_bytes(), self.request.read_bytes())
        self.assertFalse(list(self.output.glob(".pending-*")))

    def test_atomic_publication_never_exposes_partial_bytes(self):
        self.output.mkdir()
        original_link = os.link
        def inspect(source, target, **kwargs):
            self.assertFalse((self.output / target).exists())
            self.assertEqual((self.output / source).read_bytes(), b"complete")
            return original_link(source, target, **kwargs)
        with runner._directory(self.output) as directory, patch.object(os, "link", side_effect=inspect):
            runner._write_new(directory, "seal", b"complete")
        self.assertEqual((self.output / "seal").read_bytes(), b"complete")

    def test_duplicate_claims_only_one_executes(self):
        def attempt(_):
            try:
                self.run_store([4])
                return "won"
            except FileExistsError:
                return "lost"
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(sorted(pool.map(attempt, range(2))), ["lost", "won"])
        self.assertEqual(FakeStudy.calls, [(4, None)])

    def test_rust_validation_not_replaced_by_hash_validation(self):
        self.run_store()
        path = self.output / "checkpoint-2.json"
        value = json.loads(path.read_bytes())
        value["invalid_geometry"] = True
        path.write_bytes(runner._json_bytes(value))
        receipt_path = self.output / "segment-2.json"
        receipt = json.loads(receipt_path.read_bytes())
        receipt["checkpoint_sha256"] = runner._digest(path.read_bytes())
        receipt_path.write_bytes(runner._json_bytes(receipt))
        with self.assertRaisesRegex(ValueError, "Rust rejected"):
            runner.load_byte_corpus_segment(self.output, 2)

    def test_wrong_rust_cursor_never_sealed(self):
        FakeStudy.corrupt_at = 2
        with self.assertRaisesRegex(ValueError, "different cursor"):
            self.run_store()
        self.assertFalse((self.output / "segment-2.json").exists())
        self.assertEqual(runner.load_byte_corpus_segment(self.output, 0).checkpoint.completed_updates, 0)

    def test_failure_record_errors_do_not_hide_original(self):
        original = runner._write_new
        FakeStudy.fail_at = 2
        def fail(directory, name, raw):
            if name == "failed.json":
                raise OSError("failure recording also failed")
            return original(directory, name, raw)
        with patch.object(runner, "_write_new", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "simulated GPU failure"):
                self.run_store()
        self.assertEqual(runner.load_byte_corpus_segment(self.output, 0).checkpoint.completed_updates, 0)

    def test_legacy_claim_and_noncanonical_metadata_are_not_adopted(self):
        self.run_store()
        for raw in (b'{"schema":"spiraltorch.byte_corpus.segment_launch.v1"}',
                    b'{"stops": [], "stops": [4]}', b'{"seconds": NaN}', b'[]'):
            (self.output / "claim.json").write_bytes(raw)
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                runner.load_byte_corpus_segment(self.output, 2)


if __name__ == "__main__":
    unittest.main()
