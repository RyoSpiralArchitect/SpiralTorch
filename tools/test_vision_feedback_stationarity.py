"""Offline admission checks for the frozen-model probe, not a policy replica."""
import copy
import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location("stationarity", Path(__file__).with_name("probe_vision_feedback_stationarity.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
spec = importlib.util.spec_from_file_location("stationarity_verify", Path(__file__).with_name("verify_vision_feedback_stationarity.py"))
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class StationarityChecks(unittest.TestCase):
    def test_sequences_preserve_all_values_and_constant_control(self):
        sequences = probe.stationary_sequences([1., 3., 2., 2.], 2)
        self.assertEqual(sequences["recorded_order"], [1., 3., 2., 2.])
        self.assertEqual(sequences["reversed_order"], [2., 2., 3., 1.])
        self.assertEqual(sequences["constant_loss"], [2.] * 4)
        self.assertEqual(sorted(sequences["recorded_order"]), sorted(sequences["reversed_order"]))

    def test_empty_partial_and_invalid_losses_fail(self):
        for losses, count in (([], 2), ([1., 2., 3.], 2), ([1.], 0),
                              ([float("nan")], 1), ([0.], 1), ([-1.], 1)):
            with self.subTest(losses=losses, count=count), self.assertRaises(ValueError):
                probe.stationary_sequences(losses, count)

    def test_epoch_membership_must_remain_identical(self):
        rows = [dict(sample_ids=[1, 2], loss=1.), dict(sample_ids=[3, 4], loss=3.),
                dict(sample_ids=[4, 1], loss=2.), dict(sample_ids=[3, 2], loss=2.)]
        result = probe.epoch_summaries(rows, [1, 2, 3, 4], 2)
        self.assertEqual([r["mean_loss"] for r in result], [2., 2.])
        rows[-1]["sample_ids"] = [3, 4]
        with self.assertRaisesRegex(ValueError, "same samples"):
            probe.epoch_summaries(rows, [1, 2, 3, 4], 2)

    def rows(self):
        return [dict(step=i, loss=loss, applied_scale=scale, action=action,
            state_after=dict(control_step=i, observation_count=i, last_observation_step=i,
                             last_loss=loss, gate=gate, halted=halted))
            for i, (loss, scale, action, gate, halted) in enumerate([
                (2., 1., "initialize", 0., False), (1., 1., "recover", 0.5, False),
                (3., 0.75, "halt", 0., True)], 1)]

    def test_summary_counts_observed_core_actions(self):
        summary = probe.summarize_shadow(self.rows())
        self.assertEqual(summary["halted_observations"], 1)
        self.assertEqual(summary["nonidentity_controls"], 1)
        self.assertEqual(summary["adjacent_loss_increases"], 1)
        self.assertEqual(summary["actions"], {"halt": 1, "initialize": 1, "recover": 1})

    def test_changed_core_clock_loss_or_gate_fails(self):
        for key, value in (("control_step", 4), ("observation_count", 0), ("last_observation_step", 0),
                           ("last_loss", 1.), ("gate", float("nan"))):
            rows = copy.deepcopy(self.rows())
            rows[-1]["state_after"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                probe.summarize_shadow(rows)

    def frozen_rows(self):
        rows = [dict(step=i, sample_ids=[i], input_sha256=str(i) * 64,
                     loss=1., torch_loss=1., loss_bits=probe.runner.bits(1.),
                     parity=dict(max_abs_error=0., max_scaled_error=0., values=1)) for i in (1, 2)]
        return rows, copy.deepcopy(rows)

    def test_saved_inputs_and_reference_metrics_are_recomputed(self):
        rows, originals = self.frozen_rows()
        self.assertEqual(verifier.verify_rows(rows, originals, [1, 2], 2)[0]["mean_loss"], 1.)

    def test_changed_inputs_bits_or_parity_cannot_hide_in_equal_means(self):
        for key, value in (("sample_ids", [2]), ("input_sha256", "9" * 64),
                           ("loss_bits", probe.runner.bits(2.)),
                           ("parity", dict(max_abs_error=0.1, max_scaled_error=0., values=1))):
            rows, originals = self.frozen_rows()
            rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                verifier.verify_rows(rows, originals, [1, 2], 2)

    def test_missing_records_or_nonfinite_reference_fails(self):
        rows, originals = self.frozen_rows()
        with self.assertRaises(ValueError):
            verifier.verify_rows(rows[:1], originals, [1, 2], 2)
        rows[0]["torch_loss"] = float("nan")
        with self.assertRaises(ValueError):
            verifier.verify_rows(rows, originals, [1, 2], 2)


if __name__ == "__main__":
    unittest.main()
