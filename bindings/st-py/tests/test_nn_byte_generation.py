"""Installed native decoding against an independent full-forward byte oracle."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import unittest

import spiraltorch as st

HELPERS = runpy.run_path(str(Path(__file__).with_name("test_nn_byte_serving.py")))
STUDY = HELPERS["HELPERS"]


def oracle(owner, prompts, count, placeholder=255):
    """Test-only decoding from full logits, without native generation helpers."""
    windows = list(prompts)
    generated = [[] for _ in prompts]
    hashes = []
    steps = len(prompts[0])
    for _ in range(count):
        batch = st.nn.ByteLmBatch.from_windows([row + bytes([placeholder]) for row in windows])
        forward = owner.forward(owner.prepare_batch(batch))
        logits = forward.prediction_tensor().snapshot().read_values()
        last = []
        for row in range(len(prompts)):
            scores = logits[((row + 1) * steps - 1) * 256:(row + 1) * steps * 256]
            byte = max(range(256), key=scores.__getitem__)
            generated[row].append(byte)
            windows[row] = windows[row][1:] + bytes([byte])
            last.extend(scores)
        hashes.append(HELPERS["output_hash"](last))
    return generated, hashes


@unittest.skipUnless(os.getenv("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS") == "1"
                     and st.wgpu_kernel_reports_available(), "explicit WGPU runtime opt-in")
class GenerationRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        request = json.loads(STUDY["two_head_source"]())
        request["config"]["batch"] = 2
        for key in ("train_batches", "validation_batches"):
            request[key] = [batch * 2 for batch in request[key]]
        cls.request = request
        cls.study = st.nn.ByteCorpusStudy.from_json(json.dumps(request))
        cls.initial_checkpoint = cls.study.advance_wgpu(0).checkpoint
        cls.trained_checkpoint = cls.study.advance_wgpu().checkpoint
        cls.names = [request["cases"][0]["name"], request["cases"][1]["name"]]
        cls.prompts = [b"\x00\xff\x80\x01", b"rust"]

    def artifact(self, checkpoint, name):
        return self.study.case_artifact(checkpoint, name, policy=HELPERS["POLICY"])

    def test_batched_plain_and_geometry_match_independent_full_forward(self):
        for name in self.names:
            with self.subTest(name=name):
                artifact = self.artifact(self.trained_checkpoint, name)
                reference = artifact.restore_wgpu()
                generated, hashes = oracle(reference, self.prompts, 7)
                owner = artifact.restore_wgpu()
                revision = owner.attempted_updates
                report = json.loads(owner.generate_greedy(self.prompts, max_new_bytes=7))
                self.assertEqual(report["schema"], "spiraltorch.byte_corpus.generation.v1")
                self.assertEqual(report["artifact_id"], artifact.artifact_id)
                self.assertEqual(report["revision"], str(revision))
                self.assertEqual(owner.attempted_updates, revision)
                self.assertEqual(report["decoding"], "greedy_lowest_id_sliding_window.v1")
                self.assertEqual(report["window_shape"], [2, 4])
                self.assertEqual(report["prompt_sha256"], [hashlib.sha256(p).hexdigest() for p in self.prompts])
                self.assertEqual(report["generated_bytes"], generated)
                self.assertEqual(report["next_logits_f32le_sha256"], hashes)
                self.assertIsNone(report["serving"])
                self.assertNotIn("loss", report)
                self.assertEqual(json.loads(owner.generate_greedy(self.prompts, max_new_bytes=7)), report)
                self.assertEqual(owner.attempted_updates, revision)
                # Inference targets never determine the generated next byte.
                self.assertEqual(oracle(owner, self.prompts, 7, placeholder=0), (generated, hashes))
                self.assertEqual(owner.attempted_updates, revision)

    def test_invalid_inputs_leave_the_owner_usable(self):
        artifact = self.artifact(self.initial_checkpoint, self.names[1])
        owner = artifact.restore_wgpu()
        before = owner.generate_greedy(self.prompts, max_new_bytes=1)
        for count in (0, 257):
            with self.subTest(count=count), self.assertRaises(ValueError):
                owner.generate_greedy(self.prompts, max_new_bytes=count)
        for count in (True, 1.5, "2"):
            with self.subTest(count=count), self.assertRaises(TypeError):
                owner.generate_greedy(self.prompts, max_new_bytes=count)
        for count in (-1, 2 ** 100):
            with self.subTest(count=count), self.assertRaises(OverflowError):
                owner.generate_greedy(self.prompts, max_new_bytes=count)
        for prompts in ([], [b"rust"], [b"rus", b"rust"], [b"longer", b"rust"]):
            with self.subTest(prompts=prompts), self.assertRaises(ValueError):
                owner.generate_greedy(prompts, max_new_bytes=1)
        oversized = b"x" * (1024 * 1024)
        with self.assertRaises(ValueError):
            owner.generate_greedy([oversized] * 32, max_new_bytes=1)
        for prompts in (["rust", "rust"], [[1, 2, 3, 4]] * 2, [bytearray(b"rust")] * 2):
            with self.assertRaises(TypeError):
                owner.generate_greedy(prompts, max_new_bytes=1)
        self.assertEqual(owner.generate_greedy(self.prompts, max_new_bytes=1), before)

    def test_serving_switch_and_rollback_pin_generation_identity(self):
        initial = self.artifact(self.initial_checkpoint, self.names[1])
        trained = self.artifact(self.trained_checkpoint, self.names[1])
        slot = HELPERS["gate"](HELPERS["gate_value"](self.request)).bootstrap_wgpu(initial)
        before = json.loads(slot.generate_greedy(self.prompts, max_new_bytes=5, expected_generation=0))
        self.assertEqual(before["serving"], dict(generation="0", gate_sha256=slot.gate_sha256))
        for generation in (True, "0", 0.5):
            with self.assertRaises(TypeError):
                slot.generate_greedy(self.prompts, max_new_bytes=1, expected_generation=generation)
        permit = slot.prepare_candidate(trained)
        self.assertTrue(permit.accepted)
        slot.activate(permit)
        with self.assertRaisesRegex(ValueError, "stale generation dispatch"):
            slot.generate_greedy(self.prompts, max_new_bytes=5, expected_generation=0)
        after = json.loads(slot.generate_greedy(self.prompts, max_new_bytes=5, expected_generation=1))
        self.assertEqual(after["artifact_id"], trained.artifact_id)
        self.assertEqual(after["serving"]["generation"], "1")
        direct = json.loads(trained.restore_wgpu().generate_greedy(self.prompts, max_new_bytes=5))
        self.assertEqual(after["generated_bytes"], direct["generated_bytes"])
        self.assertEqual(after["next_logits_f32le_sha256"], direct["next_logits_f32le_sha256"])
        self.assertNotEqual(before["next_logits_f32le_sha256"], after["next_logits_f32le_sha256"])
        slot.rollback(1)
        with self.assertRaisesRegex(ValueError, "stale generation dispatch"):
            slot.generate_greedy(self.prompts, max_new_bytes=5, expected_generation=0)
        restored = json.loads(slot.generate_greedy(self.prompts, max_new_bytes=5, expected_generation=2))
        self.assertEqual(restored["serving"]["generation"], "2")
        restored["serving"]["generation"] = "0"
        self.assertEqual(restored, before)


if __name__ == "__main__":
    unittest.main()
