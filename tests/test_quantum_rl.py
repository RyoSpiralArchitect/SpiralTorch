import importlib.util
from pathlib import Path
import unittest


def _load_module(name: str, relative: str):
    root = Path(__file__).resolve().parent.parent
    path = root / "spiraltorch" / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
        raise RuntimeError(f"unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


qr = _load_module("spiraltorch_fallback_qr", "qr.py")
rl = _load_module("spiraltorch_fallback_rl", "rl.py")
vision = _load_module("spiraltorch_fallback_vision", "vision.py")

QuantumMeasurement = qr.QuantumMeasurement
LossStdTrigger = rl.LossStdTrigger
PolicyGradient = rl.PolicyGradient
FractalCanvas = vision.FractalCanvas


class QuantumMeasurementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.measurement = QuantumMeasurement(
            active_qubits=[0, 2],
            eta_bar=0.8,
            policy_logits=[0.2, 0.6, 0.9],
            packing_pressure=0.4,
        )

    def test_top_qubits_sorted(self) -> None:
        top = self.measurement.top_qubits(2)
        self.assertEqual(top[0][0], 2)
        self.assertGreaterEqual(top[0][1], top[1][1])

    def test_activation_density(self) -> None:
        density = self.measurement.activation_density()
        self.assertAlmostEqual(density, 2 / 3, places=6)

    def test_policy_update_contains_expected_fields(self) -> None:
        update = self.measurement.to_policy_update()
        self.assertIn("learning_rate", update)
        self.assertIn("gauge", update)
        self.assertIn("activation_density", update)
        self.assertGreater(update["learning_rate"], 1.0)
        self.assertGreater(update["gauge"], 1.0)
        self.assertAlmostEqual(update["activation_density"], 2 / 3, places=6)


class PolicyGradientQuantumTests(unittest.TestCase):
    def test_update_from_quantum_tracks_last_update(self) -> None:
        measurement = QuantumMeasurement(
            active_qubits=[1, 3],
            eta_bar=0.5,
            policy_logits=[0.1, 0.5, 0.9, 0.7],
            packing_pressure=0.3,
        )
        policy = PolicyGradient()
        policy.attach_hyper_surprise(LossStdTrigger(std_threshold=0.1, warmup=0))
        update = policy.update_from_quantum(
            measurement,
            base_rate=1.2,
            returns=[1.0, -1.0, 0.5],
            baseline=0.1,
        )
        self.assertIsNotNone(policy.last_quantum_update)
        self.assertGreater(update["learning_rate"], 1.2)
        self.assertGreater(update["gauge"], 1.2)
        stored = policy.last_quantum_update
        assert stored is not None
        self.assertAlmostEqual(update["eta_bar"], stored["eta_bar"], places=6)
        self.assertAlmostEqual(update["packing_pressure"], stored["packing_pressure"], places=6)


class FractalQuantumBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = FractalCanvas(dim=2.5)
        self.studio = qr.QuantumRealityStudio(curvature=-0.8, qubits=12)
        self.patch = self.canvas.emit_infinite_z(zoom=256.0, steps=24)

    def test_resonance_from_fractal_patch(self) -> None:
        resonance = qr.resonance_from_fractal_patch(self.patch, eta_scale=1.4)
        self.assertGreater(len(resonance.shell_weights), 0)
        self.assertGreaterEqual(resonance.eta_hint, 0.0)

    def test_quantum_measurement_from_fractal(self) -> None:
        measurement = qr.quantum_measurement_from_fractal(
            self.studio,
            self.patch,
            threshold=0.05,
            eta_scale=1.2,
        )
        self.assertIsInstance(measurement, QuantumMeasurement)
        density = measurement.activation_density()
        self.assertGreaterEqual(density, 0.0)
        update = rl.update_policy_from_fractal(
            PolicyGradient(),
            self.studio,
            self.patch,
            base_rate=1.1,
            threshold=0.05,
            eta_scale=1.2,
        )
        self.assertIn("learning_rate", update)
        self.assertIn("gauge", update)


if __name__ == "__main__":
    unittest.main()
