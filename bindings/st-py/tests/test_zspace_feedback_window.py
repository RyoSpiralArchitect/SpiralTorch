"""Exercise the native Rust loss window and emit a wasm32 replay fixture."""
import copy
import json
import os
from pathlib import Path
import struct
import unittest

import spiraltorch as st


def checkpoint(report):
    return {"config": report["config"], "state": report["state_after"]}


def advance(current, step, loss):
    control = st.zspace_optimizer_feedback_control(
        **current, target_step=step, proposed_learning_rate_scale=0.5,
    )
    observation = dict(step=step, max_steps=400, loss=loss, learning_rate=0.01)
    observed = st.zspace_optimizer_feedback_observe(**checkpoint(control), observation=observation)
    return dict(observation=observation, control=control, observed=observed)


def replay_case(width):
    initial = st.zspace_optimizer_feedback_init({"loss_window_observations": width})
    current = {key: initial[key] for key in ("config", "state")}
    records = []
    for step in range(1, 401):
        loss = struct.unpack("<f", struct.pack("<f", 0.001 + (step * 37 % 1001) / 517))[0]
        record = advance(current, step, loss)
        current = checkpoint(record["observed"])
        records.append(record)
    return dict(width=width, initial=initial, records=records, final=current)


class FeedbackWindow(unittest.TestCase):
    def test_default_wire_format_is_unchanged(self):
        implicit = st.zspace_optimizer_feedback_init()
        explicit = st.zspace_optimizer_feedback_init({"loss_window_observations": 1})
        self.assertEqual(implicit, explicit)
        self.assertNotIn("loss_window_observations", implicit["config"])
        self.assertNotIn("loss_window", implicit["state"])
        for width in (0, -1, 1.5, 2**53):
            with self.subTest(width=width), self.assertRaises((ValueError, TypeError)):
                st.zspace_optimizer_feedback_init({"loss_window_observations": width})

    def test_regression_is_detected_only_at_window_boundary(self):
        current = st.zspace_optimizer_feedback_init({
            "loss_window_observations": 2, "relative_delta_ema_alpha": 1.0,
        })
        current = {key: current[key] for key in ("config", "state")}
        for step, loss in enumerate((4., 4., 2., 2., 8., 8.), 1):
            observed = advance(current, step, loss)["observed"]
            if step % 2:
                self.assertEqual(observed["action"], "await_window")
                self.assertEqual(observed["gate_after"], current["state"]["gate"])
                self.assertIsNone(observed["relative_loss_delta"])
            if step == 4:
                self.assertGreater(observed["gate_after"], 0)
            if step == 5:
                self.assertFalse(observed["state_after"]["halted"])
            if step == 6:
                self.assertEqual(observed["action"], "halt")
                self.assertEqual(observed["relative_loss_delta"], 3.)
            current = checkpoint(observed)

    def test_missing_observations_preserve_window_and_expire_control(self):
        initial = st.zspace_optimizer_feedback_init({"loss_window_observations": 2})
        current = {key: initial[key] for key in ("config", "state")}
        for step, loss in enumerate((4., 4., 2., 2.), 1):
            current = checkpoint(advance(current, step, loss)["observed"])
        original = copy.deepcopy(current["state"])
        for step in (5, 6):
            controlled = st.zspace_optimizer_feedback_control(
                **current, target_step=step, proposed_learning_rate_scale=0.5,
            )
            current = checkpoint(controlled)
        self.assertEqual(controlled["disposition"], "stale")
        self.assertEqual(controlled["applied_learning_rate_scale"], 1.)
        self.assertEqual(current["state"]["loss_window"], original["loss_window"])
        self.assertEqual(current["state"]["observation_count"], 4)

    def test_partial_window_json_restore_and_native_fixture(self):
        cases = [replay_case(width) for width in (1, 4, 80)]
        for case in cases:
            with self.subTest(width=case["width"]):
                prefix = checkpoint(case["records"][36]["observed"])
                saved = json.loads(json.dumps(prefix, allow_nan=False))
                restored = st.zspace_optimizer_feedback_restore(**saved)
                current = {key: restored[key] for key in ("config", "state")}
                self.assertEqual(current, prefix)
                if case["width"] > 1:
                    self.assertEqual(current["state"]["loss_window"]["observations"], 37 % case["width"])
                for step, expected in enumerate(case["records"][37:], 38):
                    actual = advance(current, step, expected["observation"]["loss"])
                    self.assertEqual(actual, expected)
                    current = checkpoint(actual["observed"])
                self.assertEqual(current, case["final"])
        if destination := os.environ.get("SPIRALTORCH_FEEDBACK_WINDOW_HANDOFF"):
            with Path(destination).open("x", encoding="utf-8") as stream:
                json.dump(dict(schema="spiraltorch.feedback_window_fixture.v1", restart_at=37, cases=cases),
                          stream, allow_nan=False, separators=(",", ":"))

    def test_corruption_and_silent_reconfiguration_fail_closed(self):
        initial = st.zspace_optimizer_feedback_init({"loss_window_observations": 80})
        current = {key: initial[key] for key in ("config", "state")}
        current = checkpoint(advance(current, 1, 2.)["observed"])
        for field, value in (("observations_per_window", 4), ("observations", 0),
                             ("completed_windows", 1), ("mean", 3.), ("previous_mean", 2.)):
            bad = copy.deepcopy(current)
            bad["state"]["loss_window"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                st.zspace_optimizer_feedback_restore(**bad)
        for width in (1, 4):
            bad = copy.deepcopy(current)
            bad["config"]["loss_window_observations"] = width
            with self.subTest(width=width), self.assertRaises(ValueError):
                st.zspace_optimizer_feedback_restore(**bad)


if __name__ == "__main__":
    unittest.main()
