import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

from projection_language_fixture import weights
import rank_one_projection as run
from rank_one_projection_audit import adapter_record, effect_gate, metrics
from rank_one_roster import freeze_roster, top_rank_one


class RankOneTests(unittest.TestCase):
    def test_input_major_leading_pair_and_matched_control(self):
        w = run.W
        data = [0.0] * (w * w)
        data[3 * w + 7] = -0.8
        data[8 * w + 1] = 0.24
        dense = [math.sqrt(1 - 0.8**2 - 0.24**2)] + [0.0] * (w - 1) + data
        pair = top_rank_one(data)
        arms = run.axes(dense, pair)
        self.assertEqual(arms["rank1"][w + 3 * w + 7], -1.0)
        self.assertEqual(arms["orthogonal"][w + 3 * w + 6], 1.0)
        self.assertEqual(run.dot(arms["rank1"], arms["orthogonal"]), 0.0)
        self.assertTrue(all(abs(math.hypot(*v) - 1) < 1e-12 for v in arms.values()))
        swapped = pair | dict(left=pair["right"], right=pair["left"])
        with self.assertRaisesRegex(ValueError, "singular pair"):
            run.axes(dense, swapped)
        with self.assertRaises(ValueError):
            run.axes(dense, pair | dict(fraction_dense_fit_norm=float("nan")))
        self.assertIsNone(top_rank_one([0.0] * (w * w)))

    def test_physical_merge_edits_only_named_input_major_weight_and_bias(self):
        raw = weights()
        (n,) = struct.unpack_from("<Q", raw)
        data = json.loads(raw[8 : 8 + n])
        w = run.W
        idx = 3 * w + 7
        zero = [0.0] * (w + w * w)
        self.assertEqual(run.merge_bytes(raw, run.f32s(zero)), (raw, zero))
        zero[2], zero[w + idx] = -0.0123, 0.0153
        got, actual = run.merge_bytes(raw, run.f32s(zero))
        expected = bytearray(raw)
        for key, coord, delta in (("bias", 2, zero[2]), ("weight", idx, zero[w + idx])):
            pos = 8 + n + data["h.0.attn.c_proj." + key]["data_offsets"][0] + 4 * coord
            (old,) = struct.unpack_from("<f", raw, pos)
            struct.pack_into("<f", expected, pos, old + run.F32.unpack(run.F32.pack(delta))[0])
        self.assertEqual(bytes(expected), got)
        self.assertEqual([i for i, x in enumerate(actual) if x], [2, w + idx])
        self.assertLess(abs(math.hypot(*actual) - math.hypot(*zero)), 1e-7)
        with self.assertRaises(ValueError):
            run.merge_bytes(raw[:-1], run.f32s(zero))
        with self.assertRaises(ValueError):
            run.merge_bytes(raw, run.f32s([float("nan")] + zero[1:]))
        with self.assertRaises(ValueError):
            run.merge_bytes(raw, run.f32s(zero[:-1]))

    def test_shared_fold_direction_not_confused_with_flips_or_order_effects(self):
        rows = [r for r in run.pr.task.deck("train") if r["edge"] % 2 == 1]
        base = [[-2.0, 1.0, 0]] * 32

        def scores(h):
            return [[-2.0 + h * run.pr.sign(r), 1.0, 0] for r in rows]

        cell = metrics(rows, base, scores(-0.02), scores(0.02), 0.02)
        self.assertAlmostEqual(cell["central"], 0.02)
        self.assertTrue(cell["prediction_ok"])
        self.assertEqual(cell["two_choice_flips"], [0, 0])
        self.assertEqual(cell["full_greedy_flips"], [0, 0])
        self.assertEqual(len(cell["by_edge"]), 4)
        self.assertTrue(effect_gate(cell, cell, cell | {"central": 0.001}, 0.02))
        self.assertFalse(effect_gate(cell, cell, cell | {"central": -0.015}, 0.02))
        self.assertFalse(effect_gate(cell, cell | {"by_order": [0.05, -0.01]}, cell | {"central": 0.0}, 0.02))
        self.assertFalse(effect_gate(cell, cell | {"central": 0.017}, cell | {"central": 0.0}, 0.002))
        self.assertFalse(metrics(rows, base, scores(-0.02), scores(0.02), -0.02)["prediction_ok"])

    def test_roster_rejects_corrupt_and_incomplete_fit_only_input(self):
        with tempfile.TemporaryDirectory() as folder:
            raw, dest = (Path(folder) / k for k in ("directions", "sealed"))
            pin = run.put(raw, dict(roles={}, plan="test", fit="test"))
            for ref in (pin, "0" * 64):
                with self.assertRaises(ValueError):
                    freeze_roster(raw, ref, dest)
            self.assertFalse(dest.exists())

    def test_adapter_archive_is_bound_to_type_and_exact_serialization(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            data = b'{"base_model_identity":"model","layer":1}'
            lo = "sha256:" + run.sha(b"spiraltorch.native_lora.v1\0" + data)
            run.write(folder / (lo[7:] + ".json"), data)
            self.assertEqual(adapter_record(folder, lo, "lora")["layer"], 1)
            with self.assertRaises(ValueError):
                adapter_record(folder, lo, "geometry")
            with self.assertRaises(ValueError):
                adapter_record(folder, "sha256:" + "x" * 64, "lora")


if __name__ == "__main__":
    unittest.main()
