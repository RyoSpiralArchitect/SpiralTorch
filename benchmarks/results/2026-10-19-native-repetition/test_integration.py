"""Qualification gates with tiny native fixtures, never pretrained-model evidence."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import shutil
import struct
import tempfile
import unittest

PROBE = runpy.run_path(str(Path(__file__).with_name("integration.py")))
PROBE["PRIOR"]["PRIOR"]["PRIOR"]["forbid_ml"]()
import spiraltorch as st  # noqa: E402

FIXTURE = runpy.run_path(str(PROBE["ROOT"] / "bindings/st-py/tests/test_native_causal_lm.py"))
TRAIN = runpy.run_path(str(PROBE["TRAIN"]))["main"]
GENERATE = runpy.run_path(str(PROBE["GENERATE"]))["main"]


class IntegrationGates(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        config, weights = FIXTURE["fixture"](positions=16)
        self.model_dir = self.root / "model"
        self.model_dir.mkdir()
        header, payload = {}, bytearray()
        for name in weights.names():
            tensor = weights.tensor(name)
            values, start = tensor.values(), len(payload)
            payload.extend(struct.pack("<" + "f" * len(values), *values))
            header[name] = {"dtype": "F32", "shape": tensor.shape, "data_offsets": [start, len(payload)]}
        raw = json.dumps(header).encode()
        (self.model_dir / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
        (self.model_dir / "config.json").write_text(config)
        self.base = st.NativeCausalLm.from_gpt2(config, st.SafeTensorArchive(self.model_dir / "model.safetensors"))
        artifact = FIXTURE["artifact"](self.base, trainable=True)
        (self.root / "artifact.json").write_text(artifact.to_json())
        self.write("contract.json", {"scope": "tiny synthetic qualification test"})
        self.write("data.json", {"token_documents": [[1, 2] * 4, [3, 4] * 4]})
        self.write("prompt.json", [1, 2])
        self.recipe = json.loads(json.dumps(PROBE["TRAIN_RECIPE"]))
        self.recipe["context_length"] = 8
        self.recipe["repetition"]["config"]["candidate_source"]["proposal_top_k"] = 6
        self.write("train.json", self.recipe)
        for kind, execution in PROBE["EXECUTIONS"].items():
            recipe = json.loads(json.dumps(PROBE["GEN_RECIPE"]))
            recipe["decoding"]["top_k"] = 6
            self.write(kind + ".json", {**recipe, "execution": execution})
        self.build_training()
        for kind in PROBE["EXECUTIONS"]:
            self.build_generation(kind)

    def write(self, path, value):
        (self.root / path).write_text(json.dumps(value), encoding="utf-8")

    def pin(self, path):
        return "sha256:" + PROBE["digest"](self.root / path)

    def build_training(self, prefix="", data="data.json", recipe="train.json"):
        for arm in ("full", "part", "resume"):
            argv = ["--model-dir", str(self.model_dir), "--output", str(self.root / (prefix + "train-" + arm)),
                    "--data-json", str(self.root / data), "--save-every", "1"]
            if arm != "resume":
                argv += ["--recipe-json", str(self.root / recipe), "--artifact", str(self.root / "artifact.json"),
                         "--artifact-id", self.pin("artifact.json")]
            else:
                path = self.root / (prefix + "train-part/checkpoint-000001.json")
                argv += ["--resume", str(path), "--checkpoint-id", self.pin(path)]
            if arm == "part":
                argv += ["--stop-after", "1"]
            with contextlib.redirect_stdout(io.StringIO()):
                TRAIN(argv)

    def build_generation(self, kind, prefix="", recipe=None, prompt="prompt.json"):
        for arm in ("full", "part", "resume"):
            checkpoint = self.root / "train-full/checkpoint-000002.json"
            argv = ["--model-dir", str(self.model_dir), "--output", str(self.root / (prefix + kind + "-" + arm)),
                    "--training-checkpoint", str(checkpoint), "--training-checkpoint-id", self.pin(checkpoint)]
            if arm != "resume":
                argv += ["--recipe-json", str(self.root / (recipe or kind + ".json")),
                         "--prompt-json", str(self.root / prompt)]
            else:
                path = self.root / (prefix + kind + "-part/generation.json")
                argv += ["--resume", str(path), "--resume-id", self.pin(path)]
            if arm == "part":
                argv += ["--stop-after", "2"]
            with contextlib.redirect_stdout(io.StringIO()):
                GENERATE(argv)

    def replace_outputs(self, kind):
        for arm in ("full", "part", "resume"):
            shutil.copytree(self.root / ("other-" + kind + "-" + arm),
                            self.root / (kind + "-" + arm), dirs_exist_ok=True)

    def update_result(self, path, **changes):
        value = json.loads((self.root / path).read_text())
        value.update(changes)
        self.write(path, value)

    def assert_rejected(self):
        with self.assertRaisesRegex(ValueError, "mismatch"):
            PROBE["report"](self.root, st, self.base)

    def test_complete_receipts_pass(self):
        result = PROBE["report"](self.root, st, self.base)
        self.assertEqual(result["training"]["updates"], 2)
        self.assertEqual(set(result["generation"]), {"full_prefix", "kv_cache"})

    def test_changed_training_checkpoint_fails(self):
        self.write("train-resume/checkpoint-000002.json", {"cursor": 999})
        self.assert_rejected()

    def test_changed_generation_checkpoint_fails(self):
        self.write("kv_cache-resume/generation.json", {"cursor": 999})
        self.assert_rejected()

    def test_missing_step_fails(self):
        (self.root / "full_prefix-part/steps.jsonl").write_text("")
        self.assert_rejected()

    def test_all_empty_logs_fail(self):
        for kind in ("train", "full_prefix", "kv_cache"):
            for arm in ("full", "part", "resume"):
                (self.root / (kind + "-" + arm) / "steps.jsonl").write_text("")
        self.assert_rejected()

    def test_out_of_order_reports_fail(self):
        path = self.root / "train-full/steps.jsonl"
        path.write_text("".join(reversed(path.read_text().splitlines(keepends=True))))
        self.assert_rejected()

    def test_changed_metadata_fails(self):
        cases = [("train-full/result.json", {"completed_updates": 1}),
                 ("train-resume/result.json", {"initial": {"checkpoint_id": "wrong"}}),
                 ("kv_cache-full/result.json", {"execution": {"kind": "full_prefix"}}),
                 ("kv_cache-full/result.json", {"geometry_artifact_id": None}),
                 ("kv_cache-full/result.json", {"generated_tokens": 3}),
                 ("kv_cache-full/result.json", {"checkpoint_id": "wrong"}),
                 ("kv_cache-resume/result.json", {"initial_id": "wrong"})]
        for name, changes in cases:
            with self.subTest(changes=changes):
                original = (self.root / name).read_bytes()
                self.update_result(name, **changes)
                self.assert_rejected()
                (self.root / name).write_bytes(original)

    def test_coherent_non_repetition_training_fails(self):
        recipe = dict(self.recipe)
        recipe.pop("repetition")
        self.write("other-train.json", recipe)
        self.build_training("other-", recipe="other-train.json")
        self.replace_outputs("train")
        self.assert_rejected()

    def test_coherent_wrong_data_fails(self):
        self.write("other-data.json", {"token_documents": [[2, 3] * 4, [4, 5] * 4]})
        self.build_training("other-", data="other-data.json")
        self.replace_outputs("train")
        self.assert_rejected()

    def test_coherent_wrong_optimizer_fails(self):
        self.write("other-train.json", {**self.recipe, "max_gradient_norm": .2})
        self.build_training("other-", recipe="other-train.json")
        self.replace_outputs("train")
        self.assert_rejected()

    def test_coherent_wrong_generation_seed_fails(self):
        recipe = json.loads((self.root / "kv_cache.json").read_text())
        self.write("other-generation.json", {**recipe, "seed": 999})
        self.build_generation("kv_cache", "other-", recipe="other-generation.json")
        self.replace_outputs("kv_cache")
        self.assert_rejected()

    def test_coherent_wrong_prompt_fails(self):
        self.write("other-prompt.json", [2, 3])
        self.build_generation("kv_cache", "other-", prompt="other-prompt.json")
        self.replace_outputs("kv_cache")
        self.assert_rejected()


if __name__ == "__main__":
    unittest.main()
