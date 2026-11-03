"""Model-free protocol tests and tiny *synthetic* decoder rehearsal; never opens pretrained weights."""
import json
from pathlib import Path
import runpy
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as trial  # noqa: E402

p = trial.p


class Block:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in p.old.FORBIDDEN | {"tokenizers"}:
            raise AssertionError("foreign ML import: " + fullname)


sys.meta_path.insert(0, Block())
import spiraltorch as st  # noqa: E402


class Controls(unittest.TestCase):
    def test_frozen_split_is_disjoint_balanced_and_reproducible(self):
        rows, spec = p.freeze()
        self.assertEqual(p.pin(spec), trial.FROZEN_SHA)
        self.assertEqual(len(p.previous_keys() | {r["key"] for group in rows.values() for r in group}), 288)
        for order in p.ORDERS:
            keys = [rows["train"][i]["key"] for i in p.train_order(rows["train"], order)]
            self.assertEqual(len(set(keys)), 96)
            self.assertEqual(spec["orders"][order], p.pin(keys))
        with self.assertRaises(ValueError):
            p.train_order(rows["train"], "another-order")

    def test_origin_is_matched_including_input_and_parameter_jacobians(self):
        x, dy = [1., -2., 4., .5, 2., -1.], [.5, -.5, .25, 2., -.25, 1.]
        path = "transformer.h.1.attn"
        for arm in p.ARMS:
            artifact = p.geometry(st, "sha256:" + "a" * 64, arm, width=3, layer=1)
            runtime = st.NativeGeometryRuntime(artifact)
            batch = runtime.forward_sequence(path, x, [1, 2])
            self.assertEqual(batch.output, x)
            dx, params = batch.vjp(dy)
            self.assertEqual(dx, dy)
            for name, want in (("gate", [(.5 + 1.) / 4, (1. - .5) / 4, (1. - 1.) / 4]),
                               ("bias", [2.5 / 4, -.75 / 4, 1.25 / 4])):
                self.assertEqual(params[name], want)
            self.assertEqual(params.get("log_radius"), [0.] if arm == "wave_learned" else None)
            record = json.loads(artifact.state_json())
            next(t for t in record["placements"][0]["tensors"] if t["name"] == "bias")["values"] = [1., 0., 0.]
            nonzero = st.NativeGeometryRuntime(st.GeometryAdapterArtifact.create_json(json.dumps(record)))
            out = nonzero.forward_sequence(path, x, [1, 2]).output
            if arm == "affine":
                self.assertEqual(out[0], 1.25)
            else:
                self.assertTrue(1. < out[0] < 1.25)

    def test_synthetic_joint_updates_restore_and_score_the_same_answers(self):
        fixture = runpy.run_path(str(p.old.ROOT / "bindings/st-py/tests/test_native_causal_lm.py"))["fixture"]
        config, weights = fixture(width=4, heads=2, layers=2)
        base = st.NativeCausalLm.from_gpt2(config, weights)
        models, counts = trial.models_for(st, base, 4, 1)
        self.assertEqual(counts, [24, 24, 25])
        data = dict(keys=["a", "b", "c"], token_documents=[[1, 2, 3, 4], [4, 3, 2], [1, 2, 5, 6, 7]],
                    labels=[[-100, -100, 3, 4], [-100, -100, 2], [-100, -100, -100, 6, 7]])
        start = trial.score(st, models, data, 8, 1)
        self.assertTrue(all(c["tied"] == 3 for c in start["contrasts"].values()))
        with tempfile.TemporaryDirectory() as temp:
            loc = Path(temp) / "trial"
            restored, _ = trial.one_order(st, base, models, data, [2, 0, 1], loc, time.time() + 90, 8, 1)
            self.assertEqual(trial.read(loc / "order.json")["keys"], ["c", "a", "b"])
            self.assertEqual(len(trial.read(loc / "checkpoints.json")), 3)
            self.assertTrue(all(m.lora_artifact_id != models[0].lora_artifact_id for m in restored))
            final = trial.score(st, restored, data, 8, 1)
            self.assertTrue(all(c["target_tokens"] == 5 for c in final["contrasts"].values()))
            with self.assertRaises(FileExistsError):
                trial.one_order(st, base, models, data, [2, 0, 1], loc, time.time() + 90, 8, 1)

    def test_decision_is_paired_symmetric_and_cannot_promote_a_tiny_sign(self):
        def c(d, hits=5):
            return dict(answer_weighted_delta_nll=d, candidate_all_correct_answers=hits, reference_all_correct_answers=5)
        self.assertEqual(p.classify([c(-p.MARGIN / 2)] * 3), "inside_reporting_margin_not_equivalence")
        self.assertEqual(p.classify([c(-2 * p.MARGIN)] * 3), "fixed_wave_advantage_on_this_split")
        self.assertEqual(p.classify([c(2 * p.MARGIN)] * 3), "affine_advantage_on_this_split")
        self.assertEqual(p.classify([c(-2 * p.MARGIN)] * 2 + [c(-2 * p.MARGIN, 4)]), "mixed_or_below_gate")
        with self.assertRaises(ValueError):
            p.classify([c(-1.)] * 2)


if __name__ == "__main__":
    unittest.main()
