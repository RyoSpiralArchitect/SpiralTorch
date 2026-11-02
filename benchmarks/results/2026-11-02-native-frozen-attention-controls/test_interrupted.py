import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_interrupted import complete_lines


class InterruptedLogTests(unittest.TestCase):
    def test_only_newline_committed_records_count(self):
        self.assertEqual(
            complete_lines(b'{"update":1}\n{"update":2}'),
            ([{"update": 1}], b'{"update":2}'),
        )
        self.assertEqual(complete_lines(b'{"upd'), ([], b'{"upd'))
        self.assertEqual(complete_lines(b""), ([], b""))

    def test_corrupt_record_in_the_middle_is_not_treated_as_a_torn_tail(self):
        with self.assertRaises(ValueError):
            complete_lines(b'{"x":1}\n{bad}\n{"x"')
