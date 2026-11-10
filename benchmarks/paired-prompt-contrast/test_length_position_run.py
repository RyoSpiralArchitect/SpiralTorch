import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import length_position_run as r


class RunnerTests(unittest.TestCase):
    def test_objective_report_cannot_impersonate_plain_nll(self):
        plain = dict(steps=1, gold_nll=2., gradient_l2=[0., 1., 2.])
        smooth = dict(steps=1, objective_loss=2., label_smoothing=.1, gradient_l2=[0., 1., 2.])
        r.update_report(plain, {})
        r.update_report(smooth, dict(label_smoothing=.1))
        for report, action in ((smooth, {}), (plain, dict(label_smoothing=.1)),
                               (smooth, dict(label_smoothing=.2))):
            with self.assertRaises(ValueError):
                r.update_report(report, action)

    def test_steps_carry_state_and_do_not_repeat_first_batch(self):
        seen = []

        def call(name, lag, action, start):
            seen.append((name, action["gold"], start))
            n = start["steps"] + 1 if start else 1
            return dict(steps=n, id=str(n), serving=str(n), report=n)

        batches = [dict(prompts=[[1]], gold=[y]) for y in (2, 3, 4)]
        out = r.steps(SimpleNamespace(call=call), "test", 0, batches)
        self.assertEqual([x[1] for x in seen], [[2], [3], [4]])
        self.assertEqual([x[0] for x in seen], ["test-001", "test-002", "test-003"])
        self.assertIsNone(seen[0][2])
        self.assertEqual(seen[1][2], out[0])
        self.assertEqual(r.steps(SimpleNamespace(call=call), "resume", 0, batches[1:], out[0], out), out[1:])
        bad = out.copy()
        bad[1] = bad[1] | dict(report=-1)
        with self.assertRaisesRegex(ValueError, "resume diverged"):
            r.steps(SimpleNamespace(call=call), "bad", 0, batches[1:], out[0], bad)

    def test_first_failure_stops_schedule(self):
        seen = []

        def call(*args):
            seen.append(args)
            if len(seen) == 2:
                raise ValueError("native failure")
            return dict(steps=1)

        with self.assertRaisesRegex(ValueError, "native failure"):
            r.steps(SimpleNamespace(call=call), "test", 0, [{}, {}, {}])
        self.assertEqual(len(seen), 2)

    def test_checkpoint_binds_all_owners_and_step(self):
        owner = dict(base="base", lora="lora", geometry="geometry")
        h = json.dumps(dict(schema="spiraltorch.qkv_adamw.v1", width=32, layer=0, steps=1, **owner)).encode()
        raw = struct.pack("<I", len(h)) + h + bytes(3072 * 12)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.bin"
            p.write_bytes(raw)
            pin = "sha256:" + r.sha(p)
            r.checkpoint(p, pin, owner, 1)
            for owners, step in ((owner, 2), (owner | dict(geometry="other"), 1)):
                with self.assertRaises(ValueError):
                    r.checkpoint(p, pin, owners, step)
            p.write_bytes(raw[:-1])
            with self.assertRaises(ValueError):
                r.checkpoint(p, pin, owner, 1)

    def test_metrics_use_all_vocab_and_ties_fail(self):
        def row(values):
            bits = list(struct.unpack("<64I", struct.pack("<64f", *values)))
            return dict(ordinary=bits, cached=bits.copy())

        tied = row([0.] * 64)
        self.assertEqual(r.metric([tied], [2])["correct"], 0)
        self.assertAlmostEqual(r.metric([tied], [2])["nll"], r.math.log(64))
        v = [0.] * 64
        v[2], v[63] = 1., 2.
        self.assertEqual(r.metric([row(v)], [2])["correct"], 0)
        v[63] = 0.
        self.assertEqual(r.metric([row(v)], [2])["correct"], 1)
        v[0] = float("nan")
        with self.assertRaises(ValueError):
            r.metric([row(v)], [2])
        mismatch = row([0.] * 64)
        mismatch["cached"][0] = 1
        with self.assertRaises(ValueError):
            r.metric([mismatch], [2])

    def test_native_error_preserves_raw_failure_and_forbids_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            io = r.Native(dict(routes={"0": dict(base="b", lora="l", geometry="g")}), root,
                          root / "not-executed", root / "run", 10)
            with patch.object(r.subprocess, "run", return_value=SimpleNamespace(returncode=7)) as execute:
                with self.assertRaisesRegex(ValueError, "native error"):
                    io.call("first", 0, dict(kind="score"))
                self.assertEqual(json.loads((root / "run/first/exit.json").read_bytes()), {"code": 7})
                with self.assertRaises(FileExistsError):
                    io.call("first", 0, dict(kind="score"))
                self.assertEqual(execute.call_count, 1)


if __name__ == "__main__":
    unittest.main()
