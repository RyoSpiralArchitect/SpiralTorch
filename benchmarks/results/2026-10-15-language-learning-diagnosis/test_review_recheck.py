import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("review_recheck",Path(__file__).with_name("review_recheck.py"))
RECHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECHECK)


class SourceIdentityTests(unittest.TestCase):
    def test_checker_executes_the_captured_source(self):
        with tempfile.TemporaryDirectory() as raw:
            source = Path(raw)/"checker.py"
            before = b"VALUE = 1\n"
            source.write_bytes(before)
            module, captured = RECHECK.load_checker(source)
            source.write_bytes(b"raise RuntimeError('unexecuted revision')\n")
            self.assertEqual(module.VALUE,1)
            self.assertEqual(captured,before)
            self.assertEqual(RECHECK.sha(captured),RECHECK.sha(before))

    def test_stable_source_is_accepted(self):
        with tempfile.TemporaryDirectory() as raw:
            source = Path(raw)/"checker.py"
            source.write_bytes(b"VALUE = 1\n")
            _, captured = RECHECK.load_checker(source)
            RECHECK.require_source_unchanged(source,captured)

    def test_changed_checker_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            source = Path(raw)/"checker.py"
            source.write_bytes(b"VALUE = 1\n")
            _, captured = RECHECK.load_checker(source)
            source.write_bytes(b"VALUE = 2\n")
            with self.assertRaisesRegex(ValueError,"source changed"):
                RECHECK.require_source_unchanged(source,captured)

    def test_changed_rechecker_rejects_before_reading_inputs(self):
        original = Path.read_bytes
        def changed(path):
            return RECHECK.RECHECK_SOURCE+b"# drift\n" if path == Path(RECHECK.__file__) else original(path)
        with mock.patch.object(Path,"read_bytes",changed), \
                mock.patch.object(RECHECK.subprocess,"check_output") as git:
            with self.assertRaisesRegex(ValueError,"source changed"):
                RECHECK.recheck(Path("unused"),Path("unused"))
        git.assert_not_called()


if __name__ == "__main__":
    unittest.main()
