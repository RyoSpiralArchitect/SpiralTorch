"""Exercise the same measurement code on generated toy weights, never on the study model."""

import math
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from study import PLAN, guard_ml, masked, measure, read  # noqa: E402


class NativeMeasurement(unittest.TestCase):
    def test_selected_targets_and_greedy_trace(self):
        guard_ml()
        import spiraltorch as st

        config, weights = runpy.run_path(
            str(HERE.parents[2] / "bindings/st-py/tests/test_native_causal_lm.py")
        )["fixture"]()
        model = st.NativeCausalLm.from_gpt2(config, weights)
        ids = [1, 2, 3, 4, 5]
        tokens = {"prefix": ids[:3], "whole": ids, "labels": masked(ids[:3], ids)}
        # Use only the toy model's vocabulary and position budgets.
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(PLAN["generation"], eos_token_ids=[10], max_new_tokens=3),
            patch.dict(PLAN, evaluation_context=6),
        ):
            folder = Path(tmp)
            measure(st, model, tokens, folder, "toy")
            row = read(folder / "toy.json")
            score = row["score"]
            self.assertEqual((score["target_tokens"], score["ignored_targets"]), (2, 2))
            logits = model.forward(ids[:-1], [1, 4]).values()
            expected = []
            for pos in (2, 3):
                values = logits[11 * pos : 11 * (pos + 1)]
                maximum = max(values)
                expected.append(
                    maximum
                    + math.log(sum(math.exp(x - maximum) for x in values))
                    - values[ids[pos + 1]]
                )
            self.assertAlmostEqual(score["nll"], sum(expected) / 2, places=5)
            cfg = row["effective_config"]
            self.assertEqual(cfg["execution"], PLAN["generation"]["execution"])
            self.assertEqual(
                cfg["decoding"]["distribution"],
                PLAN["generation"]["decoding"]["distribution"],
            )
            self.assertIsNone(cfg["decoding"]["top_k"])
            self.assertEqual(cfg["decoding"]["top_p"], 1.0)
            first = model.forward(ids[:3], [1, 3], last_only=True).values()
            self.assertEqual(
                row["generated"][0], max(range(len(first)), key=first.__getitem__)
            )


if __name__ == "__main__":
    unittest.main()
