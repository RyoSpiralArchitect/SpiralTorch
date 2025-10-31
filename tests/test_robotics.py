import importlib.util
import math
from pathlib import Path
import unittest


def _load_module(name: str, relative: str):
    root = Path(__file__).resolve().parent.parent
    path = root / "spiraltorch" / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
        raise RuntimeError(f"unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "spiraltorch"
    import sys

    sys.modules[name] = module
    canonical = f"spiraltorch.{relative[:-3].replace('/', '.')}"
    sys.modules.setdefault(canonical, module)
    spec.loader.exec_module(module)
    return module


robotics = _load_module("spiraltorch_fallback_robotics", "robotics.py")

Desire = robotics.Desire
DesireLagrangianField = robotics.DesireLagrangianField
EnergyReport = robotics.EnergyReport
PsiTelemetry = robotics.PsiTelemetry
RoboticsRuntime = robotics.RoboticsRuntime
SensorFusionHub = robotics.SensorFusionHub
PolicyGradientController = robotics.PolicyGradientController


class SensorFusionHubTests(unittest.TestCase):
    def test_fuse_multiple_modalities(self) -> None:
        hub = SensorFusionHub()
        hub.register_channel("camera", 1)
        hub.register_channel("imu", 3)
        hub.calibrate("imu", bias=(0.1, 0.1, 0.1), scale=2.0)

        frame = hub.fuse({"camera": (4.0,), "imu": (0.2, 0.4, 0.3)})
        self.assertIn("camera", frame.coordinates)
        self.assertIn("imu", frame.coordinates)
        self.assertAlmostEqual(frame.coordinates["camera"][0], 4.0)
        imu_vector = frame.coordinates["imu"]
        self.assertEqual(len(imu_vector), 3)
        self.assertTrue(
            all(math.isclose(value, expected) for value, expected in zip(imu_vector, (0.2, 0.6, 0.4)))
        )

    def test_smoothing_filters_noise(self) -> None:
        hub = SensorFusionHub()
        hub.register_channel("imu", 1, smoothing=0.5)

        hub.fuse({"imu": (1.0,)})
        second = hub.fuse({"imu": (0.0,)})

        value = second.coordinates["imu"][0]
        self.assertGreater(value, 0.0)
        self.assertLess(value, 1.0)


class DesireFieldTests(unittest.TestCase):
    def test_energy_accumulates_per_channel(self) -> None:
        field = DesireLagrangianField(
            {
                "balance": Desire(target_norm=0.2, tolerance=0.05, weight=2.0),
                "power": Desire(target_norm=0.5, tolerance=0.0, weight=1.0),
            }
        )
        hub = SensorFusionHub()
        hub.register_channel("balance", 1)
        hub.register_channel("power", 1)
        frame = hub.fuse({"balance": (0.7,), "power": (0.8,)})

        energy = field.energy(frame)
        self.assertGreater(energy.total, 0.0)
        self.assertIn("balance", energy.per_channel)
        self.assertIn("power", energy.per_channel)
        self.assertGreater(energy.per_channel["balance"], energy.per_channel["power"])


class TelemetryTests(unittest.TestCase):
    def test_observe_triggers_failsafe_on_instability(self) -> None:
        telemetry = PsiTelemetry(window=4, stability_threshold=0.8, failure_energy=10.0, norm_limit=2.0)
        hub = SensorFusionHub()
        hub.register_channel("pose", 2)
        field = DesireLagrangianField({"pose": Desire(target_norm=0.0, tolerance=0.0, weight=1.0)})

        reports = []
        for payload in (
            {"pose": (0.1, 0.1)},
            {"pose": (0.5, 0.5)},
            {"pose": (1.5, 1.5)},
        ):
            frame = hub.fuse(payload)
            energy = field.energy(frame)
            report = telemetry.observe(frame, energy)
            reports.append(report)
        self.assertTrue(any(report.failsafe for report in reports))
        self.assertTrue(any("norm_overflow" in report.anomalies for report in reports))


class RoboticsRuntimeTests(unittest.TestCase):
    def test_step_integrates_policy_gradient(self) -> None:
        hub = SensorFusionHub()
        hub.register_channel("imu", 3)
        field = DesireLagrangianField({"imu": Desire(target_norm=0.0, tolerance=0.0, weight=1.0)})
        runtime = RoboticsRuntime(sensors=hub, desires=field, telemetry=PsiTelemetry())
        controller = PolicyGradientController()
        runtime.attach_policy_gradient(controller)

        result = runtime.step({"imu": (0.1, -0.2, 0.05)})
        self.assertFalse(result.halted)
        self.assertIn("learning_rate", result.commands)
        self.assertIn("gauge", result.commands)
        self.assertGreater(result.commands["learning_rate"], 0.0)

    def test_step_emits_halt_on_failsafe(self) -> None:
        hub = SensorFusionHub()
        hub.register_channel("imu", 3)
        field = DesireLagrangianField({"imu": Desire(target_norm=0.0, tolerance=0.0, weight=1.0)})
        telemetry = PsiTelemetry(window=2, stability_threshold=0.2, failure_energy=0.01, norm_limit=0.2)
        runtime = RoboticsRuntime(sensors=hub, desires=field, telemetry=telemetry)

        result = runtime.step({"imu": (1.0, 1.0, 1.0)})
        self.assertTrue(result.halted)
        self.assertIn("halt", result.commands)


if __name__ == "__main__":
    unittest.main()
