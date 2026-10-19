"""Publication refuses incomplete or failed measurements; no ML dependencies."""
import copy
import hashlib
import json
from pathlib import Path
import runpy
import unittest

q = runpy.run_path(str(Path(__file__).with_name("qualify.py")))
retain = runpy.run_path(str(Path(__file__).with_name("retain_failure.py")))["failure_receipt"]


def fixture():
    names = [f"{owner}-{offset}-{kind}" for owner in ["base", "learned"]
             for offset in [0, 128, 512] for kind in q["CONFIGS"]]
    cases, identities, hashes = {}, {}, {"cached": {}, "old_full_prefix": {}}
    for name in names:
        owner, offset, kind = name.split("-")
        prompt = list(range(int(offset), int(offset)+8))
        identity = {"owner": owner, "offset": int(offset), "kind": kind, "prompt": prompt,
                    "config": q["CONFIGS"][kind], "geometry": q["PRIOR"]["GEOMETRY"] if owner == "learned" else None}
        identities[name] = identity
        def checkpoint(tokens):
            payload = json.dumps({"schema": "spiraltorch.native_causal_lm_generation.v1",
                "rng": "rand_chacha_0_3_chacha20_seed_u64_top53.v1", "base_model_identity": q["BASE"],
                "geometry_artifact_id": identity["geometry"], "config": identity["config"],
                "prompt_length": 8, "tokens": tokens})
            return {"payload": payload, "pin": "sha256:" + hashlib.sha256(payload.encode()).hexdigest()}
        cases[name] = {"owner": owner, "offset": int(offset), "kind": kind,
            "generated": [1]*8, "prompt": prompt, "tokens": prompt+[1]*8,
            "steps": [{"step": i, "token_id": 1} for i in range(1, 9)],
            "mid": checkpoint(prompt+[1]*3), **checkpoint(prompt+[1]*8)}
        hashes["cached"][name] = hashlib.sha256(json.dumps(cases[name]).encode()).hexdigest()
        hashes["old_full_prefix"][name] = hashlib.sha256(("old-"+name).encode()).hexdigest()
    metric = {"max_logit_abs": 1e-5, "argmax_exact": True, "within_gate": True}
    teacher = [{"owner": owner, "length": length,
                "steps": [{"step": i, "cache_values": 123, **metric} for i in range(9)]}
               for owner in ["base", "learned"] for length in q["LENGTHS"]]
    resumed = [{"mode": mode, "case": name, "before_exact": True, "steps_exact": True,
                "checkpoint_exact": True, "continued_tokens": 5, "source_sha256": hashes[mode][name]}
               for mode in ["cached", "old_full_prefix"] for name in names]
    timing = {"torch": "test", "transformers": "test", "python": "test", "platform": "test",
              "torch_threads": 2, "torch_interop_threads": 1, "extension_sha256": "a"*64,
              "rows": [{"arm": arm, "length": length, "round": i, "warmup": i < 2,
                         "prefill_ms": .1, "decode_ms": .2, "total_ms": .3, **metric}
                        for arm in q["ARMS"] for length in q["TIMING_LENGTHS"] for i in range(9)]}
    return cases, teacher, resumed, timing, identities, hashes, {name: 5 for name in names}


class PublicationTests(unittest.TestCase):
    def test_complete_fixture(self):
        q["validate"](*fixture())

    def test_missing_or_duplicate_conditions(self):
        for index in (1, 2, 3):
            for duplicate in (False, True):
                args = fixture()
                rows = args[index]["rows"] if index == 3 else args[index]
                rows.pop()
                if duplicate:
                    rows.append(copy.deepcopy(rows[0]))
                with self.assertRaises(ValueError):
                    q["validate"](*args)

    def test_false_or_nonfinite_metrics(self):
        for key, value in [("max_logit_abs", float("nan")), ("max_logit_abs", float("inf")),
                           ("max_logit_abs", True), ("max_logit_abs", .1),
                           ("argmax_exact", False), ("within_gate", False),
                           ("decode_ms", 0), ("total_ms", .4), ("warmup", False)]:
            args = fixture()
            args[3]["rows"][0][key] = value
            with self.assertRaises(ValueError):
                q["validate"](*args)

    def test_incomplete_teacher_and_resume(self):
        args = fixture()
        args[1][0]["steps"].pop()
        with self.assertRaises(ValueError):
            q["validate"](*args)
        for key, value in [("checkpoint_exact", False), ("before_exact", False),
                           ("steps_exact", False), ("continued_tokens", 4)]:
            args = fixture()
            args[2][0][key] = value
            with self.assertRaises(ValueError):
                q["validate"](*args)

    def test_changed_pin_and_step_identity(self):
        for key, value in [("pin", "sha256:wrong"), ("tokens", [3])]:
            args = fixture()
            next(iter(args[0].values()))[key] = value
            with self.assertRaises(ValueError):
                q["validate"](*args)

    def test_retention_cannot_turn_failure_into_pass_or_hide_bad_rows(self):
        with self.assertRaises(ValueError):
            retain(*fixture())
        args = fixture()
        args[3]["rows"][0].update(max_logit_abs=.001, within_gate=False)
        result = retain(*args)
        self.assertEqual(result["status"], "UNQUALIFIED_LOGIT_GATE_FAILED")
        self.assertEqual(len(result["failed_rows"]), 1)
        for key, value in [("max_logit_abs", float("nan")), ("within_gate", True)]:
            changed = copy.deepcopy(args)
            changed[3]["rows"][-1][key] = value
            if key == "within_gate":
                changed[3]["rows"][-1]["max_logit_abs"] = .1
            with self.assertRaises(ValueError):
                retain(*changed)

    def test_checkpoint_case_midpoint_and_recipe_are_bound(self):
        for mutate in ("body", "midpoint", "owner", "recipe"):
            args = fixture()
            case = next(iter(args[0].values()))
            if mutate == "body":
                case["payload"] = "test"
            elif mutate == "midpoint":
                case["mid"] = {"payload": case["payload"], "pin": case["pin"]}
            elif mutate == "owner":
                case["owner"] = "learned"
            else:
                state = json.loads(case["payload"])
                state["config"]["seed"] = 123456
                case["payload"] = json.dumps(state)
            case["pin"] = "sha256:" + hashlib.sha256(case["payload"].encode()).hexdigest()
            with self.assertRaises(ValueError):
                q["validate"](*args)

    def test_both_resume_modes_bind_source_and_complete_length(self):
        for mode in ("cached", "old_full_prefix"):
            for key, value in [("source_sha256", "stale"), ("continued_tokens", 4)]:
                args = fixture()
                next(r for r in args[2] if r["mode"] == mode)[key] = value
                with self.assertRaises(ValueError):
                    q["validate"](*args)

    def test_private_debug_fields_cannot_leak_even_after_an_earlier_failure(self):
        for target in ("teacher", "step", "resume", "timing", "row"):
            args = fixture()
            args[3]["rows"][0].update(max_logit_abs=.1, within_gate=False)
            row = {"teacher": args[1][-1], "step": args[1][-1]["steps"][-1],
                   "resume": args[2][-1], "timing": args[3], "row": args[3]["rows"][-1]}[target]
            row["private_tokens"] = [1, 2, 3]
            with self.assertRaises(ValueError):
                retain(*args)

    def test_teacher_failure_is_not_labeled_native_pass(self):
        args = fixture()
        args[1][0]["steps"][0].update(max_logit_abs=.01, within_gate=False)
        result = retain(*args)
        self.assertFalse(result["teacher_all_pass"])
        self.assertEqual(result["status"], "UNQUALIFIED_LOGIT_GATE_FAILED")

    def test_nested_metadata_and_invalid_scalar_formats_are_rejected(self):
        for key, value in [("python", {"private_tokens": [1, 2]}), ("platform", [1, 2]),
                           ("extension_sha256", "invalid"), ("torch_interop_threads", True)]:
            args = fixture()
            args[3]["rows"][0].update(max_logit_abs=.1, within_gate=False)
            args[3][key] = value
            with self.assertRaises(ValueError):
                retain(*args)
        runtime = {"python": "3.12.6", "platform": "test", "extension_sha256": "a"*64,
                   "optional_ml_imports_forbidden": True}
        q["validate_runtime"](runtime, native=True)
        for key, value in [("python", {"private_tokens": [1, 2]}), ("extension_sha256", "not-hex"),
                           ("optional_ml_imports_forbidden", 1), ("optional_ml_imports_forbidden", False)]:
            with self.assertRaises(ValueError):
                q["validate_runtime"]({**runtime, key: value}, native=True)


if __name__ == "__main__":
    unittest.main()
