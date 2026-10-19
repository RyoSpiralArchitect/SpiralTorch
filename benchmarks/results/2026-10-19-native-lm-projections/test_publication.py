"""Aggregate export rejects missing conditions, invented flags and private fields."""
import copy
import json
from pathlib import Path
import runpy
import unittest

P = runpy.run_path(str(Path(__file__).with_name("publish.py")))
Q = P["Q"]


def timing():
    return {"contract": Q["contract"](), "extension_sha256": "a"*64, "torch": "test",
        "transformers": "test", "python": "test", "platform": "test",
        "torch_threads": 2, "torch_interop_threads": 1,
        "accuracy": [{"length": n, "arm": arm, "step": i, "max_logit_abs": .0001,
                     "argmax_exact": True, "within_gate": True}
                    for n in Q["LENGTHS"] for arm in Q["ARMS"] for i in range(Q["APPENDS"]+1)],
        "rows": [{"length": n, "arm": arm, "round": i, "warmup": i < Q["WARMUPS"],
                  "prefill_ms": .1, "decode_ms": .2, "total_ms": .3,
                  "max_logit_abs": .0007 if arm.startswith("f32") and n == 64 else .0001,
                  "argmax_exact": True, "within_gate": not (arm.startswith("f32") and n == 64)}
                 for n in Q["LENGTHS"] for arm in Q["ARMS"] for i in range(Q["WARMUPS"] + Q["REPEATS"])]}


def native():
    rows = []
    for policy in Q["POLICIES"]:
        for owner in ("base", "learned"):
            row = {"policy": policy, "owner": owner,
                "projections": [{"length": n, "step": i, "full_sha256": "b"*64, "cache_sha256": "b"*64,
                    "max_logit_abs": 0., "argmax_exact": True, "within_gate": True}
                    for n in Q["LENGTHS"] for i in range(Q["APPENDS"]+1)],
                "generation": [{"recipe": name, "steps_sha256": "c"*64, "checkpoint_id": "sha256:"+"d"*64,
                    "tokens": 8} for name in Q["OLD"]["CONFIGS"]]}
            if owner == "learned":
                row["training"] = {"steps_sha256": "e"*64, "checkpoint_id": "sha256:"+"f"*64,
                    "updates": 2, "initial_loss": 4.5, "second_loss": 4.6}
            rows.append(row)
    return {"contract": Q["contract"](), "extension_sha256": "a"*64, "python": "test", "platform": "test",
        "foreign_imports_forbidden": True, "baseline_sha256": "b"*64, "f32_bitwise_compatible": True, "rows": rows}


class PublicationTests(unittest.TestCase):
    def test_all_prefix_accuracy_required_and_affects_status(self):
        for action in ("missing", "duplicate", "unknown"):
            record = timing()
            record["accuracy"].pop()
            if action != "missing":
                record["accuracy"].append(copy.deepcopy(record["accuracy"][0]))
            if action == "unknown":
                record["accuracy"][-1]["step"] = 999
            with self.assertRaises(ValueError):
                P["timing"](record)
        record = timing()
        row = next(r for r in record["accuracy"] if r["arm"] == "f64_cache" and r["step"] == 0)
        row.update(max_logit_abs=.001, within_gate=False)
        P["timing"](record)
        self.assertEqual(P["status"](True, True, True, record["rows"] + record["accuracy"])[0], "UNQUALIFIED")
        summary = next(r for r in P["summaries"](record) if r["arm"] == "f64_cache" and r["length"] == row["length"])
        self.assertFalse(summary["all_logit_gates_pass"])
        self.assertEqual(summary["max_logit_abs"], .001)

    def test_saved_seals_must_match_records_and_current_helpers(self):
        before, after = native(), native()
        seal = Q["contract"]()
        P["validate_seals"](before, after, seal, seal)
        for index in range(4):
            args = [copy.deepcopy(before), copy.deepcopy(after), copy.deepcopy(seal), copy.deepcopy(seal)]
            target = args[index]["contract"] if index < 2 else args[index]
            helper = next(iter(target["helper_sha256"]))
            target["helper_sha256"][helper] = "0"*64
            with self.assertRaises(ValueError):
                P["validate_seals"](*args)

    def test_control_status_is_derived_and_metadata_strings_never_export(self):
        record = timing()
        self.assertEqual(P["status"](True, True, True, record["rows"])[0],
                         "SCOPED_WIDE_READOUT_PASS_WITH_F32_FAILING_CONTROL")
        for row in record["rows"]:
            row.update(max_logit_abs=.0001, within_gate=True)
        self.assertEqual(P["status"](True, True, True, record["rows"]), ("SCOPED_READOUT_PASS", True))
        for fixture in (record, native()):
            for key in ("python", "platform", "torch", "transformers"):
                fixture[key] = "[17, 42, 99]"
            exported = P["public_runtime"](fixture)
            self.assertNotIn("[17, 42, 99]", json.dumps(exported))
            self.assertTrue(set(exported).isdisjoint({"python", "platform", "torch", "transformers"}))

    def test_helper_closure_is_part_of_protocol(self):
        self.assertEqual(set(Q["contract"]()["helper_sha256"]), {
            "benchmarks/results/2026-10-18-native-lm-cache/qualify.py",
            "benchmarks/results/2026-10-18-native-lm-generation/qualify.py",
            "benchmarks/results/2026-10-18-native-corpus-training/qualify.py",
            "benchmarks/results/2026-10-18-native-gpt2-learning/probe.py",
            "bindings/st-py/examples/native_geometry_corpus_training.py"})

    def test_complete_inputs_keep_known_f32_failures(self):
        P["timing"](timing())
        P["native"](native())
        old = native()
        old["rows"] = old["rows"][:2]
        del old["baseline_sha256"], old["f32_bitwise_compatible"]
        P["native"](old, baseline=True)

    def test_missing_duplicate_and_unknown_timing_conditions(self):
        for action in ("missing", "duplicate", "unknown"):
            value = timing()
            value["rows"].pop()
            if action != "missing":
                value["rows"].append(copy.deepcopy(value["rows"][0]))
            if action == "unknown":
                value["rows"][-1]["arm"] = "selected_fastest"
            with self.assertRaises(ValueError):
                P["timing"](value)

    def test_gate_flags_nonfinite_and_duration_accounting(self):
        for key, value in (("max_logit_abs", float("nan")), ("max_logit_abs", float("inf")),
                           ("max_logit_abs", True), ("within_gate", False), ("argmax_exact", 1),
                           ("total_ms", 0.4), ("decode_ms", 0), ("warmup", False), ("round", False)):
            record = timing()
            record["rows"][0][key] = value
            with self.assertRaises(ValueError):
                P["timing"](record)
        record = timing()
        next(row for row in record["rows"] if not row["within_gate"])["within_gate"] = True
        with self.assertRaises(ValueError):
            P["timing"](record)

    def test_private_payload_fields_are_never_copied(self):
        for where in ("root", "row", "projection", "generation", "training"):
            record = native()
            target = {"root": record, "row": record["rows"][0],
                "projection": record["rows"][0]["projections"][0],
                "generation": record["rows"][0]["generation"][0],
                "training": record["rows"][1]["training"]}[where]
            target["private_tokens"] = [1, 2, 3]
            with self.assertRaises(ValueError):
                P["native"](record)
        record = timing()
        record["rows"][-1]["raw_logits"] = [0.]
        with self.assertRaises(ValueError):
            P["timing"](record)

    def test_scalar_metadata_and_identity_formats_are_checked(self):
        for key, value in (("extension_sha256", "bad"), ("platform", {"tokens": [1]}),
                           ("foreign_imports_forbidden", 1), ("f32_bitwise_compatible", "yes")):
            record = native()
            record[key] = value
            with self.assertRaises(ValueError):
                P["native"](record)
        record = timing()
        record["torch_threads"] = True
        with self.assertRaises(ValueError):
            P["timing"](record)

    def test_missing_native_conditions_rejected(self):
        for where in ("rows", "projections", "generation"):
            record = native()
            (record["rows"] if where == "rows" else record["rows"][0][where]).pop()
            with self.assertRaises(ValueError):
                P["native"](record)


if __name__ == "__main__":
    unittest.main()
