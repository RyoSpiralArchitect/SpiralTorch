import itertools
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matrix as m
from admit_tokenizer import verify as verify_tokenizer


def encode(text):
    return list(text.encode())


class Instrument:
    def __init__(self, winners):
        self.calls = []
        self.winners = iter(winners)

    def next_token_contrast(self, ids, target, alternative):
        self.calls.append((tuple(ids), target, alternative))
        margin = float(len(self.calls))
        return dict(greedy_id=next(self.winners), target_id=target, alternative_id=alternative,
                    target_logit=margin, alternative_logit=0., target_minus_alternative=margin)


class MatrixTests(unittest.TestCase):
    def test_token_admission_rejects_an_unpinned_file_before_loading_a_tokenizer(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "tokenizer.json"
            path.write_bytes(b"{}")
            with self.assertRaisesRegex(ValueError, "wrong tokenizer"):
                verify_tokenizer(path)

    def test_roster_and_display_are_paired_and_counterbalanced(self):
        rows = m.matrix()
        spec = m.manifest(rows)
        self.assertEqual((spec["records"], spec["pairs"], spec["prompts"]), (32, 96, 192))
        self.assertEqual(spec["pair_sha256"], "afab2502944ffbbf8a1031142dc7d54a1d5f72659122ce356b340ad9642f8c73")
        self.assertEqual(Counter(p.first_form for p in rows), {"record": 48, "question": 48})
        by_key = defaultdict(list)
        for p in rows:
            by_key[p.key].append(p)
            self.assertEqual(p.answer, " " + dict(p.display)[p.field])
            self.assertEqual(p.question.split("\n")[0], " | ".join(f"{f}: {v}" for f, v in p.display))
            self.assertEqual(p.record.split("\n")[0], "Record: " + "; ".join(f"{f} = {v}" for f, v in p.display) + ".")
        for ps in by_key.values():
            self.assertEqual(len({p.display for p in ps}), 1)
            self.assertEqual({p.field for p in ps}, {"color", "shape", "material"})

    def test_an_old_key_cannot_be_swapped_back_in(self):
        data = json.loads(m.MANIFEST.read_bytes())
        recent = {k for rows in data["lengths"].values() for k, *_ in rows}
        universe = {"-".join(map(str, x)) for x in itertools.product(range(8), repeat=3)}
        reused = min(universe - set(m.FRESH) - recent)
        with patch.object(m, "FRESH", [reused, *m.FRESH[1:]]), self.assertRaisesRegex(ValueError, "older studies"):
            m.matrix()

    def test_only_prefixes_reach_the_model_in_declared_order(self):
        rows = m.matrix()
        for form in ("record", "question"):
            pair = next(r for r in rows if r.first_form == form)
            names = (form, "question" if form == "record" else "record")
            model = Instrument([10, 42])
            result = m.inspect_pair(pair, encode, model)
            self.assertEqual(model.calls, [(tuple(encode(getattr(pair, name))), 32, 10) for name in names])
            self.assertEqual([result["forms"][n]["winner"] for n in names], ["LF", "other"])
            self.assertEqual(result["record_minus_question_margin"], -1. if form == "record" else 1.)
            target = m.inspect_pair(pair, encode, Instrument([32, 32]))
            self.assertTrue(all(s["winner"] == "target" for s in target["forms"].values()))

    def test_validate_both_token_boundaries_before_any_forward(self):
        p = m.matrix()[0]
        for text, replacement in [(p.record + p.answer, [42]), (p.record + p.answer, encode(p.record) + [9]),
                                  (p.question + "\n", encode(p.question) + [10, 10]),
                                  (p.record + p.answer, encode(p.record) + [10])]:
            model = Instrument([])
            with self.subTest(text=text), self.assertRaises(ValueError):
                m.inspect_pair(p, lambda s: replacement if s == text else encode(s), model)
            self.assertEqual(model.calls, [])

    def test_readout_weights_records_not_prompt_count_and_rejects_incomplete_pairs(self):
        pairs = m.matrix()
        scores = [m.inspect_pair(p, encode, Instrument([10, 42])) for p in pairs]
        out = m.summarize(pairs, reversed(scores))
        self.assertEqual((out["records"], out["pairs"], len(out["record_means"])), (32, 96, 32))
        self.assertEqual(out["mean_record_delta"], 0.)
        self.assertEqual(set(out["record_means"].values()), {-1/3, 1/3})
        self.assertEqual((out["transition_pairs"]["LF"]["other"], out["transition_pairs"]["other"]["LF"]), (48, 48))
        for invalid in (scores[:-1], [*scores, scores[0]],
                        [{**scores[0], "record_minus_question_margin": 99}, *scores[1:]],
                        [{**scores[0], "field": scores[1]["field"]}, {**scores[1], "field": scores[0]["field"]}, *scores[2:]]):
            with self.assertRaises(ValueError):
                m.summarize(pairs, invalid)


if __name__ == "__main__":
    unittest.main()
