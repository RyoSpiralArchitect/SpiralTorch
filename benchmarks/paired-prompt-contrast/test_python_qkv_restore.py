"""Model-free tests of transport/equality gates, not native numerical evidence."""
import hashlib
import io
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import python_qkv_restore as w


def raw(ids):
    return struct.pack("<64I", *([sum(ids)] * 64))


class Prefix:
    serving_id = "snapshot"

    def __init__(self, ids, calls):
        self.token_ids = list(ids)
        self.cache_values = len(ids) * 16
        self.calls = calls

    def logits_f32le(self):
        return raw(self.token_ids)

    def branch(self, token):
        self.calls.append(("branch", token))
        return Prefix(self.token_ids + [token], self.calls)


class Trainer:
    serving_id = "snapshot"

    def __init__(self, corrupt=False):
        self.calls = []
        self.corrupt = corrupt

    def prefill(self, ids, *, max_cache_values):
        self.calls.append(("prefill", list(ids)))
        return Prefix(ids, self.calls)

    def last_logits_f32le(self, ids):
        self.calls.append(("ordinary", list(ids)))
        return bytes(256) if self.corrupt else raw(ids)


class WitnessTests(unittest.TestCase):
    def data(self):
        cell = dict(prompts=[[1, 2], [1, 3]], prefix_len=1)
        saved = dict(serving="snapshot", rows=[dict(ordinary=[sum(q)]*64, cached=[sum(q)]*64)
                                               for q in cell["prompts"]])
        return cell, saved

    def test_shared_prefix_and_exact_raw_streams(self):
        trainer = Trainer()
        a, b = io.BytesIO(), io.BytesIO()
        result = w.compare_cell(trainer, *self.data(), a, b, lambda: None)
        expected = raw([1, 2]) + raw([1, 3])
        self.assertEqual(a.getvalue(), expected)
        self.assertEqual(b.getvalue(), expected)
        self.assertEqual(result, dict(rows=2, prefills=1, branches=2,
                                     logits_sha256=hashlib.sha256(expected).hexdigest()))
        self.assertEqual(trainer.calls, [("prefill", [1]), ("branch", 2), ("ordinary", [1, 2]),
                                         ("branch", 3), ("ordinary", [1, 3])])

    def test_first_difference_stops_and_keeps_failing_raw_row(self):
        trainer = Trainer(corrupt=True)
        a, b = io.BytesIO(), io.BytesIO()
        with self.assertRaisesRegex(ValueError, "row 0 mismatch"):
            w.compare_cell(trainer, *self.data(), a, b, lambda: None)
        self.assertEqual(a.getvalue(), bytes(256))
        self.assertEqual(b.getvalue(), raw([1, 2]))
        self.assertEqual(len(trainer.calls), 3)

    def test_bad_saved_owner_or_count_precedes_execution(self):
        for field, value in (("serving", "another"), ("rows", [])):
            trainer = Trainer()
            cell, saved = self.data()
            saved[field] = value
            with self.assertRaises(ValueError):
                w.compare_cell(trainer, cell, saved, io.BytesIO(), io.BytesIO(), lambda: None)
            self.assertEqual(trainer.calls, [])

    def test_guard_precedes_native_work(self):
        trainer = Trainer()
        def stop():
            raise TimeoutError("budget")
        with self.assertRaises(TimeoutError):
            w.compare_cell(trainer, *self.data(), io.BytesIO(), io.BytesIO(), stop)
        self.assertEqual(trainer.calls, [])

    def test_reference_rejects_boolean_overflow_or_different_cache(self):
        for value in (True, -1, 2**32, 1.0):
            with self.assertRaises(ValueError):
                w.reference(dict(ordinary=[value]*64, cached=[value]*64))
        with self.assertRaises(ValueError):
            w.reference(dict(ordinary=[1]*64, cached=[2]*64))

    def test_input_hash_budget_and_exclusive_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input"
            pin = w.put(path, {"a": 1})
            self.assertTrue(w.checked(path, pin))
            with self.assertRaisesRegex(ValueError, "byte budget"):
                w.checked(path, pin, 1)
            with self.assertRaisesRegex(ValueError, "changed input"):
                w.checked(path, "0"*64)
            with self.assertRaises(FileExistsError):
                w.put(path, {"a": 2})
            self.assertEqual(w.sha(path), pin)

    def test_forbidden_import_attempt_is_recorded(self):
        finder = w.RejectForeignML()
        with self.assertRaises(RuntimeError):
            finder.find_spec("torch.nn")
        self.assertEqual(finder.attempts, ["torch.nn"])
        self.assertIsNone(finder.find_spec("json"))


if __name__ == "__main__":
    unittest.main()
