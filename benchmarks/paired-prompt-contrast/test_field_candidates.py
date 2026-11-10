from pathlib import Path
import math
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from field_candidates import FIELDS, FieldVocabulary, RequestRouter, rank_support


class FieldCandidateTests(unittest.TestCase):
    def setUp(self):
        words = ["\u010a", "What", "\u0120is", "\u0120the", "?", "Return", ".", "Answer", ":", *("\u0120"+f for f in FIELDS)]
        self.vocab = dict(zip(words, range(len(words))))
        self.router = RequestRouter.gpt2(self.vocab, 32)

    def test_routes_only_the_request_even_when_body_contains_other_fields_or_answers(self):
        for field, suffix in self.router.rules:
            class OpaqueRecord:
                def __len__(self): return len(suffix) + 11
                def __getitem__(self, i):
                    if i < 11: raise AssertionError("read record contents")
                    return suffix[i-11]
            self.assertEqual(self.router.match(OpaqueRecord()), (field, suffix))
            self.assertEqual(self.router.match([27, 20, 30, *suffix]), (field, suffix))

    def test_missing_truncated_or_ambiguous_request_is_not_guessed(self):
        f, suffix = self.router.rules[0]
        for prompt in (suffix[1:], (*suffix, 1), (*suffix[:-1], False), (), [20, 3]):
            with self.assertRaises(ValueError): self.router.match(prompt)
        rules = list(self.router.rules)
        rules[2] = (rules[2][0], suffix)
        with self.assertRaisesRegex(ValueError, "ambiguous"): RequestRouter(32, tuple(rules))
        with self.assertRaises(ValueError): RequestRouter.gpt2({}, 32)

    def test_fit_deduplicates_within_field_and_allows_shared_first_tokens(self):
        data = [("color", 2), ("color", 0), ("color", 2), ("shape", 3), ("material", 2)]
        v = FieldVocabulary.fit(data, 8)
        self.assertEqual(v.support("color"), (0,2))
        self.assertEqual(v.support("shape"), (3,))
        self.assertEqual(v.support("material"), (2,))
        self.assertEqual(v.pooled(), (0,2,3))
        self.assertEqual(v, FieldVocabulary.fit(reversed(data), 8))
        self.assertNotIn(1, v.support("color"))  # no inference-time fill for an unseen answer
        for bad in ("unknown", None):
            with self.assertRaises(ValueError): v.support(bad)
        for d in (data[:-1], [*data, ("color", True)], [*data, ("field", 1)], [*data, ("shape", 8)]):
            with self.assertRaises(ValueError): FieldVocabulary.fit(d, 8)

    def test_constrained_choice_is_label_free_and_separate_from_global_greedy(self):
        raw = struct.pack("<6f", 0, 20, 3, -2, 3, 2)
        self.assertEqual(rank_support(raw, (4,2,0), 6), (2,4,0))
        self.assertEqual(rank_support(raw, (4,2,1), 6), (1,2,4))
        for ids in ((), (False,), (0,0), (6,), (-1,), range(99)):
            with self.assertRaises(ValueError): rank_support(raw, ids, 6)
        with self.assertRaises(ValueError): rank_support(raw[:-1], (1,), 6)
        with self.assertRaises(ValueError): rank_support(struct.pack("<2f", 1, math.inf), (1,), 2)


if __name__ == "__main__":
    unittest.main()
