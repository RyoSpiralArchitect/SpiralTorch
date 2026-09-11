"""The host handoff and all acceptance policy live in Rust, including CPU builds."""
import json
import os
from pathlib import Path
import sys
import unittest
import spiraltorch as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from resident_module_handoff_fixture import model, flat, SHAPE, INPUT, TARGET


def changed(base):
    payload = json.loads(base.to_json())
    for p in payload["parameters"]:
        p["values"] = [v + 0.125 for v in p["values"]]
    return st.nn.InferencePlan.from_json(json.dumps(payload))


class Surface(unittest.TestCase):
    def test_all_roles_fusion_drift_and_explicit_reset(self):
        host = model()
        base = host.inference_plan(SHAPE)
        updated = changed(base).fuse_pointwise()
        host.attach_hypergrad(-1., 0.01)
        host.attach_realgrad(0.01)
        with self.assertRaisesRegex(ValueError, "optimizer state"):
            base.apply_parameters_to(host, updated)
        self.assertEqual(host.inference_plan(SHAPE).to_json(), base.to_json())
        for invalid in ("", "auto", "preserve", "RESET", None, True):
            with self.assertRaises((TypeError, ValueError)):
                base.apply_parameters_to(host, updated, optimizer_state=invalid)
        self.assertEqual(base.apply_parameters_to(host, updated, optimizer_state="reset"), 6)
        current = host.inference_plan(SHAPE)
        self.assertEqual(json.loads(current.to_json())["parameters"], json.loads(updated.to_json())["parameters"])
        self.assertEqual(current.apply_parameters_to(host, current), 6)  # No parameter-local optimizer state remains.
        with self.assertRaisesRegex(ValueError, "changed since baseline"):
            base.apply_parameters_to(host, updated)

    def test_bad_structure_and_values_leave_model_and_tapes_intact(self):
        host = model()
        base = host.inference_plan(SHAPE)
        host.attach_hypergrad(-1., 0.01)
        payload = json.loads(changed(base).to_json())
        payload["stages"][0]["gelu"] = False
        bad = st.nn.InferencePlan.from_json(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "structure"):
            base.apply_parameters_to(host, bad, optimizer_state="reset")
        self.assertEqual(base.to_json(), host.inference_plan(SHAPE).to_json())
        with self.assertRaisesRegex(ValueError, "optimizer state"):
            base.apply_parameters_to(host, changed(base))
        payload = json.loads(base.to_json())
        payload["parameters"][-1]["values"][0] = float("nan")
        with self.assertRaises(ValueError):
            st.nn.InferencePlan.from_json(json.dumps(payload))
        self.assertEqual(base.to_json(), host.inference_plan(SHAPE).to_json())

    def test_pending_zero_gradients_and_duplicate_names_are_rejected(self):
        host = model()
        base = host.inference_plan(SHAPE)
        host.backward(st.Tensor(4,2,INPUT), st.Tensor(4,2,[0.]*8))
        with self.assertRaisesRegex(ValueError, "optimizer state"):
            base.apply_parameters_to(host, changed(base))
        duplicate = st.nn.Sequential()
        duplicate.add(st.nn.Linear("same", 2, 2))
        duplicate.add(st.nn.Linear("same", 2, 2))
        base = duplicate.inference_plan(SHAPE)
        with self.assertRaisesRegex(ValueError, "unique"):
            base.apply_parameters_to(duplicate, base)


@unittest.skipUnless(os.environ.get("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS") == "1", "real WGPU opt-in")
class Gpu(unittest.TestCase):
    def test_native_training_returns_to_original_model_and_new_trainer_phase(self):
        host = model()
        base = host.inference_plan(SHAPE)
        x = st.Tensor(4,2,INPUT)
        original = flat(host.forward(x))
        gpu = base.fuse_pointwise().compile_graph_training_wgpu(gradient_policy="module_compatible")
        self.assertNotEqual(gpu.adapter_info()["device_type"], "Cpu")
        gpu.upload_batch_values(INPUT,TARGET)
        for _ in range(16):
            gpu.step(0.05)
            gpu.loss_snapshot().read()
        gpu.step(0.)
        state = gpu.state_snapshot().read_state()
        updated = st.nn.InferencePlan.from_json(state.to_plan().to_json())
        del gpu
        self.assertEqual(base.apply_parameters_to(host, updated), 6)
        result = flat(host.forward(x))
        self.assertNotEqual(result,original)
        for a,b in zip(result,state.prediction_values()): self.assertAlmostEqual(a,b,places=5)
        trainer = st.nn.ModuleTrainer(backend="cpu")
        trainer.prepare(host)
        before = host.inference_plan(SHAPE).to_json()
        host.backward(x, st.Tensor(4,2,[0.25]*8))
        trainer.step(host)
        self.assertNotEqual(before,host.inference_plan(SHAPE).to_json())


if __name__ == "__main__": unittest.main()
