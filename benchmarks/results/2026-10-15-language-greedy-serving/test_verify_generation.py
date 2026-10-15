"""Fault controls for the qualifier, not a replacement for native execution."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

FILE = Path(__file__).with_name("verify_generation.py")
SPEC = importlib.util.spec_from_file_location("generation_probe", FILE)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class ProbeControls(unittest.TestCase):
    def exercise(self, switch):
        class Slot:
            artifact_id = "incumbent"
            generation = 0

            def tensor_device(self):
                return NS(adapter_info=lambda: {"device_type": "fake"})

            def generate_greedy(self, prompts, *, max_new_bytes, expected_generation):
                if self.generation != expected_generation:
                    raise ValueError("stale generation dispatch")
                return json.dumps(dict(artifact_id=self.artifact_id,
                    serving=dict(generation=str(self.generation), gate_sha256="gate"),
                    prompt_sha256=[PROBE.digest(p) for p in prompts],
                    generated_bytes=[[32]], next_logits_f32le_sha256=["hash"]))

            def prepare_candidate(self, artifact):
                return NS(accepted=True)

            def activate(self, candidate):
                self.generation = 1
                if switch:
                    self.artifact_id = "candidate"

            def rollback(self, expected_generation):
                self.generation = 2
                self.artifact_id = "incumbent"

        gate = NS(batch_shape=(1, 2), request_sha256="gate", bootstrap_wgpu=lambda _: Slot())
        fake = NS(nn=NS(ByteCorpusServingGate=NS(from_json=lambda _: gate),
                       ByteCorpusModelArtifact=NS(from_json=lambda raw, expected_id: expected_id)))
        payload = dict(gate_json="fixture", windows=[1, 2, 3],
                       incumbent=dict(json="fixture", id="incumbent"),
                       candidate=dict(json="fixture", id="candidate"))
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "private.json"
            source.write_text(json.dumps(payload))
            with patch.dict(sys.modules, spiraltorch=fake), patch.object(
                    PROBE, "oracle", return_value=([[32]], ["hash"])):
                return PROBE.run(source, 1)

    def test_explicit_artifact_ids_reject_counter_only_activation(self):
        with self.assertRaisesRegex(ValueError, "selected artifact identity"):
            self.exercise(False)

    def test_valid_fixture_still_passes(self):
        self.assertTrue(self.exercise(True)["selected_artifact_ids_verified"])

    def test_optimized_python_stops_before_input_or_native_import(self):
        run = subprocess.run([sys.executable, "-I", "-S", "-B", "-O", str(FILE)],
                             capture_output=True, text=True, check=False)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("must not run under optimized Python", run.stderr)


if __name__ == "__main__":
    unittest.main()
