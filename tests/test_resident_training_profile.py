import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from profile_resident_graph_training import validate


class ProfileAdmission(unittest.TestCase):
    def fixture(self):
        tick = 2**60 + 1
        profile = dict(
            schema="spiraltorch.graph_training_gpu_profile.v1",
            accepted=True, instrumented=True, batch_generation="1",
            timing_complete=True, ambiguous_zero_pairs=0,
            timestamp_period_ns=2., zero_intervals=1, gpu_span_ns=8.,
            passes=[
                dict(phase="forward_mixed", sample_status="observed", start_tick=str(tick), end_tick=str(tick), elapsed_ns=0.),
                dict(phase="update", sample_status="observed", start_tick=str(tick+1), end_tick=str(tick+4), elapsed_ns=6.),
            ],
            phase_totals_ns=dict(forward_mixed=0., update=6.),
        )
        return dict(status="passed", guards=dict(rollback=True), controls=[{} for _ in range(12)],
                    profiles=[dict(copy.deepcopy(profile), warmup=i < 3, submitted_step=str(i+1)) for i in range(12)])

    def test_exact_integer_ticks_and_zero_intervals_remain_valid(self):
        validate(self.fixture())

    def test_no_acceptance_or_counter_shortcuts(self):
        for field, bad in [("accepted", False), ("instrumented", False), ("submitted_step", 1),
                           ("timing_complete", False), ("ambiguous_zero_pairs", 1),
                           ("batch_generation", "2"), ("warmup", False), ("timestamp_period_ns", 0),
                           ("zero_intervals", 0), ("gpu_span_ns", 6.)]:
            with self.subTest(field=field):
                row = self.fixture()
                row["profiles"][0][field] = bad
                with self.assertRaises(ValueError):
                    validate(row)

    def test_no_float_ticks_reversed_intervals_or_invented_durations(self):
        for field, bad in [("start_tick", 2**60), ("start_tick", str(2**64)),
                           ("sample_status", "ambiguous_zero_pair"),
                           ("end_tick", "0"), ("elapsed_ns", 10.), ("elapsed_ns", float("nan"))]:
            with self.subTest(field=field):
                row = self.fixture()
                row["profiles"][0]["passes"][0][field] = bad
                with self.assertRaises(ValueError):
                    validate(row)

    def test_no_missing_guards_samples_or_phase_totals(self):
        for mutate in [lambda r: r["guards"].clear(), lambda r: r["profiles"].pop(),
                       lambda r: r["controls"].pop(),
                       lambda r: r["profiles"][0]["phase_totals_ns"].update(update=1.)]:
            row = self.fixture()
            mutate(row)
            with self.assertRaises(ValueError):
                validate(row)


if __name__ == "__main__":
    unittest.main()
