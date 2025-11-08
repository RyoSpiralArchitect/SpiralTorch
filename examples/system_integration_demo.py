#!/usr/bin/env python3
"""
SpiralTorch 0.3.0 - System Integration Example

This example demonstrates how SpiralTorch components connect organically
as a unified system, showing the flow of data and transformations across:
- Tensor creation and interop
- Spiral consensus analytics
- Robotics-inspired geometry and telemetry
- Vision processing
- Reinforcement learning
- System-wide feedback loops

This is a complete end-to-end workflow showing SpiralTorch as a "giant OS"
for machine learning.
"""

from __future__ import annotations

import importlib.util
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _import_from_repo(module_name: str, relative_path: str):
    """Load a module directly from the repository if the package shim is absent."""

    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(module_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

try:
    import spiraltorch as st
    print("✓ SpiralTorch 0.3.0 loaded")
except ImportError as exc:  # pragma: no cover - script entry guard
    print(f"✗ Failed to import spiraltorch: {exc}")
    sys.exit(1)

try:  # pragma: no branch - prefer package import, fallback to repository source
    from spiraltorch.rl import LossStdTrigger, PolicyGradient
except ModuleNotFoundError:
    rl_module = _import_from_repo("spiraltorch.rl", "spiraltorch/rl.py")
    LossStdTrigger = rl_module.LossStdTrigger
    PolicyGradient = rl_module.PolicyGradient

try:
    from spiraltorch.robotics import (
        Desire,
        DesireLagrangianField,
        DriftSafetyPlugin,
        EnergyReport,
        FusedFrame,
        GravityField,
        GravityWell,
        PsiTelemetry,
        SafetyReview,
        SensorFusionHub,
        TelemetryReport,
        ZSpaceDynamics,
        ZSpaceGeometry,
    )
except ModuleNotFoundError:
    robotics_module = _import_from_repo("spiraltorch.robotics", "spiraltorch/robotics.py")
    Desire = robotics_module.Desire
    DesireLagrangianField = robotics_module.DesireLagrangianField
    DriftSafetyPlugin = robotics_module.DriftSafetyPlugin
    EnergyReport = robotics_module.EnergyReport
    FusedFrame = robotics_module.FusedFrame
    GravityField = robotics_module.GravityField
    GravityWell = robotics_module.GravityWell
    PsiTelemetry = robotics_module.PsiTelemetry
    SafetyReview = robotics_module.SafetyReview
    SensorFusionHub = robotics_module.SensorFusionHub
    TelemetryReport = robotics_module.TelemetryReport
    ZSpaceDynamics = robotics_module.ZSpaceDynamics
    ZSpaceGeometry = robotics_module.ZSpaceGeometry

try:
    from spiraltorch.vision import FractalCanvas
except ModuleNotFoundError:
    vision_module = _import_from_repo("spiraltorch.vision", "spiraltorch/vision.py")
    FractalCanvas = vision_module.FractalCanvas


def _seed_spiraltorch(seed: int) -> None:
    """Seed SpiralTorch if available and keep Python determinism aligned."""

    if hasattr(st, "set_global_seed"):
        st.set_global_seed(seed)
    random.seed(seed)


def _safe_mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = _safe_mean(values)
    variance = _safe_mean([(value - mean) ** 2 for value in values])
    return math.sqrt(variance)


class IntegratedMLPipeline:
    """An integrated ML pipeline demonstrating organic system connections."""

    def __init__(self, z_dim: int = 4, learning_rate: float = 0.02, curvature: float = -0.9):
        """Initialize the integrated pipeline."""

        print("\n" + "=" * 70)
        print("INITIALIZING INTEGRATED ML PIPELINE")
        print("=" * 70)

        self.z_dim = int(z_dim)
        self.base_learning_rate = float(learning_rate)
        self.curvature = float(curvature)

        _seed_spiraltorch(42)
        print("✓ Deterministic seed established (Python + SpiralTorch where available)")

        self.geometry = ZSpaceGeometry.non_euclidean(self.curvature)
        self.gravity = GravityField()
        self.gravity.add_well(
            "features",
            GravityWell.relativistic(mass=max(self.z_dim, 1), speed_of_light=3.5),
        )
        self.dynamics = ZSpaceDynamics(geometry=self.geometry, gravity=self.gravity)

        self.desire_field = DesireLagrangianField(
            desires={
                "features": Desire(target_norm=1.2, tolerance=0.35, weight=0.7),
                "labels": Desire(target_norm=0.6, tolerance=0.25, weight=0.4),
            },
            dynamics=self.dynamics,
        )

        self.sensor_fusion = SensorFusionHub()
        self.telemetry = PsiTelemetry(window=6, stability_threshold=0.6, geometry=self.geometry)

        self.fractal_canvas = FractalCanvas(dim=float(self.z_dim))
        self.policy = PolicyGradient()
        self.trigger = LossStdTrigger(std_threshold=0.05, decay=0.92, warmup=2)
        self.policy.attach_hyper_surprise(self.trigger)

        self.safety = DriftSafetyPlugin(word_name="SpiralOps", hazard_cut=0.85)
        self.channel_smoothing = {"features": 0.65, "labels": 0.45}

        self.weights: list[float] = []
        self.feature_axis: st.Axis | None = None
        self.label_dim: int | None = None

        self.metrics_history: list[dict[str, float]] = []
        self.policy_history: list[dict[str, float]] = []
        self.step_count = 0

        print(
            "✓ Pipeline ready "
            f"(z_dim={self.z_dim}, base_lr={self.base_learning_rate:.4f}, κ={self.curvature})"
        )

    # ------------------------------------------------------------------
    # Core processing stages
    # ------------------------------------------------------------------
    def ingest_data(
        self,
        data: Sequence[Sequence[float]],
        labels: Sequence[Sequence[float]],
    ) -> tuple[st.Tensor, st.Tensor]:
        """Ingest training data with semantic labeling."""

        if not data:
            raise ValueError("data batch must contain at least one sample")
        if not labels:
            raise ValueError("label batch must contain at least one sample")

        print("\n" + "-" * 70)
        print("STEP 1: DATA INGESTION")
        print("-" * 70)

        from spiraltorch import Axis

        batch_size = len(data)
        feature_dim = len(data[0])
        label_dim = len(labels[0])

        if any(len(sample) != feature_dim for sample in data):
            raise ValueError("all data samples must share the same feature dimension")
        if any(len(sample) != label_dim for sample in labels):
            raise ValueError("all label samples must share the same label dimension")

        batch_axis = Axis("batch", batch_size)
        feature_axis = Axis("feature", feature_dim)

        data_tensor = st.tensor(data, axes=[batch_axis, feature_axis])
        label_tensor = st.tensor(labels, axes=[batch_axis, Axis("label", label_dim)])

        if self.feature_axis is None:
            self.feature_axis = feature_axis
            feature_smoothing = self.channel_smoothing["features"]
            self.sensor_fusion.register_channel(
                "features", dimension=feature_dim, smoothing=feature_smoothing
            )
            self.weights = [0.0 for _ in range(feature_dim)]
            self.safety.set_threshold("features", 1.2 * feature_dim)
        if self.label_dim is None:
            self.label_dim = label_dim
            label_smoothing = self.channel_smoothing["labels"]
            self.sensor_fusion.register_channel(
                "labels",
                dimension=label_dim,
                smoothing=label_smoothing,
                optional=False,
                max_staleness=3.0,
            )
            self.sensor_fusion.calibrate("labels", scale=0.85)
            self.safety.set_threshold("labels", 0.9 * label_dim + 0.5)

        print(f"✓ Ingested {batch_size} samples with {feature_dim} features and {label_dim} labels")
        print(f"  Data axes: {data_tensor.axis_names()}")
        print(f"  Data shape: {data_tensor.shape}")

        return data_tensor, label_tensor

    def analyze_spiral_consensus(
        self, data_tensor: st.Tensor | "st.LabeledTensor"
    ) -> tuple[dict[str, float], list[list[float]], list[list[float]]]:
        """Compute spiral consensus metrics from the data tensor."""

        print("\n" + "-" * 70)
        print("STEP 2: SPIRAL CONSENSUS ANALYSIS")
        print("-" * 70)

        base_tensor = getattr(data_tensor, "tensor", data_tensor)
        soft, hard, spiral, metrics = base_tensor.row_softmax_hardmax_spiral()
        print(
            "✓ Spiral consensus metrics "
            f"(entropy={metrics.get('mean_entropy', 0.0):.4f}, "
            f"coherence={metrics.get('spiral_coherence', 0.0):.4f})"
        )

        return metrics, soft.tolist(), spiral.tolist()

    def compute_robotics_feedback(
        self,
        data: Sequence[Sequence[float]],
        labels: Sequence[Sequence[float]],
    ) -> tuple[EnergyReport, TelemetryReport, FusedFrame]:
        """Fuse sensor data to compute energy and stability feedback."""

        print("\n" + "-" * 70)
        print("STEP 3: ROBOTICS TELEMETRY")
        print("-" * 70)

        feature_dim = len(data[0])
        label_dim = len(labels[0])

        feature_mean = [
            sum(sample[idx] for sample in data) / len(data)
            for idx in range(feature_dim)
        ]
        label_mean = [
            sum(sample[idx] for sample in labels) / len(labels)
            for idx in range(label_dim)
        ]

        frame = self.sensor_fusion.fuse({"features": feature_mean, "labels": label_mean})
        energy_report = self.desire_field.energy(frame)
        telemetry_report = self.telemetry.observe(frame, energy_report)

        anomaly_text = ", ".join(telemetry_report.anomalies) or "none"
        print(
            "✓ Telemetry -> energy={:.4f}, stability={:.4f}, failsafe={}, anomalies={}".format(
                energy_report.total,
                telemetry_report.stability,
                telemetry_report.failsafe,
                anomaly_text,
            )
        )

        return energy_report, telemetry_report, frame

    def review_safety(
        self,
        frame: FusedFrame,
        energy_report: EnergyReport,
        telemetry_report: TelemetryReport,
    ) -> SafetyReview:
        """Evaluate safety posture using the drift plugin."""

        print("\n" + "-" * 70)
        print("STEP 4: SAFETY REVIEW")
        print("-" * 70)

        review = self.safety.review(frame, energy_report, telemetry_report)
        flagged = ", ".join(review.flagged_frames) or "none"
        print(
            "✓ Safety review -> hazard_total={:.4f}, refused={}, flagged={}".format(
                review.hazard_total,
                review.refused,
                flagged,
            )
        )

        return review

    def generate_fractal_context(self) -> tuple[dict[str, float], float]:
        """Emit a fractal patch to inform learning rate modulation."""

        print("\n" + "-" * 70)
        print("STEP 5: FRACTAL VISION CONTEXT")
        print("-" * 70)

        patch = self.fractal_canvas.emit_zspace_infinite(dim=float(self.z_dim))
        eta = patch.eta_bar()
        density_mean = _safe_mean(patch.density)
        support_min, support_max = patch.support

        summary = {
            "dimension": patch.dimension,
            "eta": eta,
            "density_mean": density_mean,
            "support_min": support_min,
            "support_max": support_max,
        }

        print(
            "✓ Fractal patch -> "
            f"dim={patch.dimension:.2f}, η̄={eta:.4f}, density_mean={density_mean:.4f}, "
            f"support=[{support_min:.4f}, {support_max:.4f}]"
        )

        return summary, eta

    def reinforce_with_policy(
        self,
        returns: Sequence[float],
        baseline: float,
        geometry_feedback: dict[str, float],
        eta: float,
    ) -> tuple[float, dict[str, float]]:
        """Blend telemetry with policy gradient signals for adaptation."""

        print("\n" + "-" * 70)
        print("STEP 6: POLICY GRADIENT FEEDBACK")
        print("-" * 70)

        self.policy.attach_geometry_feedback(geometry_feedback)
        self.trigger.geometry_eta = eta
        self.trigger.geometry_curvature = self.curvature

        update = self.policy.step(returns, baseline=baseline)
        lr_scale = max(0.1, min(2.5, update.get("learning_rate", 1.0)))
        gauge_scale = max(0.1, min(2.5, update.get("gauge", 1.0)))
        effective_lr = self.base_learning_rate * lr_scale

        summary = {
            "lr_scale": lr_scale,
            "gauge_scale": gauge_scale,
            "effective_learning_rate": effective_lr,
        }

        print(
            "✓ Policy update -> "
            f"lr_scale={lr_scale:.4f}, gauge_scale={gauge_scale:.4f}, "
            f"effective_lr={effective_lr:.5f}"
        )

        self.policy_history.append(dict(summary))
        return effective_lr, summary

    def update_weights(
        self,
        data: Sequence[Sequence[float]],
        errors: Sequence[float],
        soft_rows: Sequence[Sequence[float]],
        spiral_rows: Sequence[Sequence[float]],
        effective_lr: float,
        gauge_scale: float,
    ) -> None:
        """Update the pipeline weights using consensus-informed gradients."""

        print("\n" + "-" * 70)
        print("STEP 7: CONSENSUS WEIGHT UPDATE")
        print("-" * 70)

        if not self.weights:
            return

        reg = 0.05 * gauge_scale
        for row_idx, spiral_row in enumerate(spiral_rows):
            soft_row = soft_rows[row_idx]
            gradient_base = errors[row_idx % len(errors)]
            for col_idx, spiral_value in enumerate(spiral_row):
                consensus_delta = spiral_value - soft_row[col_idx]
                feature = data[row_idx][col_idx]
                gradient = gradient_base * feature + consensus_delta
                updated = self.weights[col_idx] * (1.0 - reg) + effective_lr * gradient
                self.weights[col_idx] = updated

        print(
            "✓ Updated weights -> "
            + ", ".join(f"w[{i}]={value:.4f}" for i, value in enumerate(self.weights))
        )

    def harmonize_system_state(
        self,
        telemetry_report: TelemetryReport,
        fractal_summary: dict[str, float],
        safety_review: SafetyReview,
    ) -> dict[str, float]:
        """Adapt base parameters using telemetry, vision, and safety context."""

        print("\n" + "-" * 70)
        print("STEP 8: SYSTEM HARMONIZATION")
        print("-" * 70)

        hazard_relief = 1.0 / (1.0 + max(safety_review.hazard_total, 0.0))
        stability = max(0.0, min(1.0, telemetry_report.stability))
        eta = fractal_summary.get("eta", 0.0)
        density = fractal_summary.get("density_mean", 0.0)
        alignment = 1.0 / (1.0 + abs(eta - density))
        harmony = 0.4 * stability + 0.35 * hazard_relief + 0.25 * alignment

        target_lr = 0.008 + harmony * 0.018
        self.base_learning_rate = 0.85 * self.base_learning_rate + 0.15 * target_lr

        curvature_shift = math.tanh((eta - 0.5) * 1.5) * 0.05
        updated_curvature = self.curvature + curvature_shift
        updated_curvature = min(-0.15, max(-1.8, updated_curvature))
        if abs(updated_curvature - self.curvature) > 1e-6:
            self.curvature = updated_curvature
            self.geometry = ZSpaceGeometry.non_euclidean(self.curvature)
            self.dynamics.geometry = self.geometry
            self.desire_field.set_dynamics(self.dynamics)
            self.telemetry.set_geometry(self.geometry)

        feature_target = max(0.2, min(0.95, 0.35 + harmony * 0.45))
        label_target = max(0.2, min(0.9, 0.3 + harmony * 0.35))
        self.channel_smoothing["features"] = feature_target
        self.channel_smoothing["labels"] = label_target
        if self.feature_axis is not None:
            self.sensor_fusion.configure_smoothing("features", feature_target)
        if self.label_dim is not None:
            self.sensor_fusion.configure_smoothing("labels", label_target)

        summary = {
            "system_harmony": harmony,
            "hazard_relief": hazard_relief,
            "base_learning_rate": self.base_learning_rate,
            "curvature": self.curvature,
            "feature_smoothing": feature_target,
            "label_smoothing": label_target,
        }

        print(
            "✓ Harmonized -> harmony={:.4f}, base_lr={:.5f}, curvature={:.3f}, "
            "feature_smoothing={:.3f}, label_smoothing={:.3f}".format(
                harmony,
                self.base_learning_rate,
                self.curvature,
                feature_target,
                label_target,
            )
        )

        return summary

    def project_future_state(
        self,
        current_loss: float,
        effective_lr: float,
        harmony: float,
        stability: float,
    ) -> dict[str, float]:
        """Forecast short-term evolution for narrative context."""

        print("\n" + "-" * 70)
        print("STEP 9: FUTURE PROJECTION")
        print("-" * 70)

        recent_losses = [entry["loss"] for entry in self.metrics_history[-3:]]
        recent_losses.append(current_loss)
        if len(recent_losses) >= 2:
            slope = (recent_losses[-1] - recent_losses[0]) / max(len(recent_losses) - 1, 1)
        else:
            slope = 0.0
        forecast_loss = max(0.0, current_loss + slope)
        projected_lr = effective_lr * (0.9 + harmony * 0.2)
        adaptation_readiness = max(0.0, min(1.0, (harmony + stability) / 2.0))

        print(
            "✓ Projection -> forecast_loss={:.6f}, projected_lr={:.6f}, readiness={:.4f}".format(
                forecast_loss,
                projected_lr,
                adaptation_readiness,
            )
        )

        return {
            "projected_loss": forecast_loss,
            "projected_effective_lr": projected_lr,
            "adaptation_readiness": adaptation_readiness,
        }

    # ------------------------------------------------------------------
    # Training orchestration
    # ------------------------------------------------------------------
    def train_step(
        self,
        data: Sequence[Sequence[float]],
        labels: Sequence[Sequence[float]],
    ) -> dict[str, float]:
        """Execute one training step through the integrated pipeline."""

        print("\n" + "=" * 70)
        print(f"TRAINING STEP {self.step_count}")
        print("=" * 70)

        data_tensor, label_tensor = self.ingest_data(data, labels)
        consensus_metrics, soft_rows, spiral_rows = self.analyze_spiral_consensus(data_tensor)
        energy_report, telemetry_report, frame = self.compute_robotics_feedback(data, labels)
        safety_review = self.review_safety(frame, energy_report, telemetry_report)
        fractal_summary, eta = self.generate_fractal_context()

        returns = [
            float(consensus_metrics.get("spiral_coherence", 0.0)),
            float(consensus_metrics.get("average_enrichment", 0.0)),
            telemetry_report.stability,
            eta,
        ]
        baseline = float(consensus_metrics.get("mean_entropy", 0.0))
        geometry_feedback = {
            "min_learning_rate_scale": 1.0 + max(fractal_summary["eta"], 0.0),
            "max_learning_rate_scale": 1.0 + max(fractal_summary["density_mean"], 0.0),
        }
        effective_lr, policy_summary = self.reinforce_with_policy(
            returns, baseline, geometry_feedback, eta
        )

        predictions: list[float] = []
        target_signals: list[float] = []
        for sample, target in zip(data, labels):
            prediction = sum(w * x for w, x in zip(self.weights, sample))
            predictions.append(prediction)
            target_signals.append(_safe_mean(target))
        errors = [target - pred for target, pred in zip(target_signals, predictions)]
        loss = _safe_mean([error * error for error in errors])
        error_std = _safe_std(errors)

        self.update_weights(
            data,
            errors,
            soft_rows,
            spiral_rows,
            effective_lr,
            policy_summary["gauge_scale"],
        )

        weights_norm = sum(abs(value) for value in self.weights)
        geometry_norms = [self.geometry.metric_norm(sample) for sample in data]
        avg_geometry_norm = _safe_mean(geometry_norms)
        gravity_potential = [
            self.gravity.potential("features", max(norm, 1e-6)) or 0.0
            for norm in geometry_norms
        ]
        avg_potential = _safe_mean(gravity_potential)

        harmony_summary = self.harmonize_system_state(telemetry_report, fractal_summary, safety_review)
        projection_summary = self.project_future_state(
            loss,
            policy_summary["effective_learning_rate"],
            harmony_summary["system_harmony"],
            telemetry_report.stability,
        )

        step_metrics = {
            "step": float(self.step_count),
            "loss": float(loss),
            "error_std": float(error_std),
            "effective_learning_rate": policy_summary["effective_learning_rate"],
            "lr_scale": policy_summary["lr_scale"],
            "gauge_scale": policy_summary["gauge_scale"],
            "spiral_coherence": float(consensus_metrics.get("spiral_coherence", 0.0)),
            "average_enrichment": float(consensus_metrics.get("average_enrichment", 0.0)),
            "mean_entropy": float(consensus_metrics.get("mean_entropy", 0.0)),
            "energy_total": float(energy_report.total),
            "stability": float(telemetry_report.stability),
            "gravitational_energy": float(energy_report.gravitational),
            "failsafe_flag": float(telemetry_report.failsafe),
            "anomaly_count": float(len(telemetry_report.anomalies)),
            "safety_hazard_total": float(safety_review.hazard_total),
            "safety_refused": float(safety_review.refused),
            "safety_flagged": float(len(safety_review.flagged_frames)),
            "fractal_eta": float(fractal_summary["eta"]),
            "fractal_density_mean": float(fractal_summary["density_mean"]),
            "avg_geometry_norm": float(avg_geometry_norm),
            "avg_gravity_potential": float(avg_potential),
            "weights_norm": float(weights_norm),
            "system_harmony": float(harmony_summary["system_harmony"]),
            "hazard_relief": float(harmony_summary["hazard_relief"]),
            "base_learning_rate": float(harmony_summary["base_learning_rate"]),
            "curvature": float(harmony_summary["curvature"]),
            "feature_smoothing": float(harmony_summary["feature_smoothing"]),
            "label_smoothing": float(harmony_summary["label_smoothing"]),
            "projected_loss": float(projection_summary["projected_loss"]),
            "projected_effective_lr": float(projection_summary["projected_effective_lr"]),
            "adaptation_readiness": float(projection_summary["adaptation_readiness"]),
            "timestamp": float(time.time()),
        }

        self.metrics_history.append(step_metrics)

        print("\n" + "-" * 70)
        print("STEP TELEMETRY")
        print("-" * 70)
        for key, value in step_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")

        self.step_count += 1
        return step_metrics

    def summarize(self) -> None:
        """Print pipeline summary and statistics."""

        print("\n" + "=" * 70)
        print("PIPELINE SUMMARY")
        print("=" * 70)

        if not self.metrics_history:
            print("No training steps executed")
            return

        total_steps = len(self.metrics_history)
        print(f"Total steps: {total_steps}")

        def collect(key: str) -> list[float]:
            return [entry[key] for entry in self.metrics_history if key in entry]

        avg_loss = _safe_mean(collect("loss"))
        avg_lr = _safe_mean(collect("effective_learning_rate"))
        avg_coherence = _safe_mean(collect("spiral_coherence"))
        avg_eta = _safe_mean(collect("fractal_eta"))
        avg_stability = _safe_mean(collect("stability"))
        avg_energy = _safe_mean(collect("energy_total"))
        avg_hazard = _safe_mean(collect("safety_hazard_total"))
        avg_harmony = _safe_mean(collect("system_harmony"))
        avg_readiness = _safe_mean(collect("adaptation_readiness"))

        print(f"Average loss: {avg_loss:.6f}")
        print(f"Average effective LR: {avg_lr:.6f}")
        print(f"Mean spiral coherence: {avg_coherence:.6f}")
        print(f"Mean fractal η̄: {avg_eta:.6f}")
        print(f"Average stability: {avg_stability:.6f}")
        print(f"Average energy: {avg_energy:.6f}")
        print(f"Average safety hazard: {avg_hazard:.6f}")
        print(f"Mean system harmony: {avg_harmony:.6f}")
        print(f"Average adaptation readiness: {avg_readiness:.6f}")

        if len(self.metrics_history) >= 2:
            initial = self.metrics_history[0]["loss"]
            final = self.metrics_history[-1]["loss"]
            improvement = ((initial - final) / max(abs(initial), 1e-9)) * 100.0
            print(f"Loss improvement: {improvement:.2f}%")

        print("\nPolicy history (learning rate scales):")
        for idx, update in enumerate(self.policy_history):
            print(
                f"  Step {idx}: lr_scale={update['lr_scale']:.4f}, "
                f"gauge_scale={update['gauge_scale']:.4f}, "
                f"effective_lr={update['effective_learning_rate']:.6f}"
            )

        print("\nFinal weights:")
        if self.weights:
            print("  " + ", ".join(f"w[{i}]={value:.6f}" for i, value in enumerate(self.weights)))
        else:
            print("  (Weights unavailable)")

        print("\nFinal harmonized parameters:")
        print(f"  Base learning rate: {self.base_learning_rate:.6f}")
        print(f"  Curvature: {self.curvature:.6f}")
        print(
            "  Channel smoothing -> features={:.4f}, labels={:.4f}".format(
                self.channel_smoothing["features"],
                self.channel_smoothing["labels"],
            )
        )


def main() -> None:
    """Run the integrated pipeline demonstration."""

    print("\n" + "=" * 70)
    print("SPIRALTORCH 0.3.0 - SYSTEM INTEGRATION EXAMPLE")
    print("=" * 70)
    print("\nDemonstrating organic system connections:")
    print("  • Data ingestion with semantic labeling")
    print("  • Spiral consensus analytics for telemetry generation")
    print("  • Robotics-inspired geometry and stability monitoring")
    print("  • Safety drift review guiding harmonization")
    print("  • Fractal vision context for geometry coherence")
    print("  • Policy gradient reinforcement steering learning")
    print("  • System harmonization and future projection for adaptation")
    print("  • System-wide feedback and adaptation")

    pipeline = IntegratedMLPipeline(z_dim=4, learning_rate=0.02, curvature=-0.9)

    training_data = [
        ([0.0, 0.0, 0.0, 1.0], [0.0, 1.0]),
        ([0.0, 1.0, 0.0, 0.0], [1.0, 0.0]),
        ([1.0, 0.0, 1.0, 0.0], [1.0, 0.0]),
        ([1.0, 1.0, 0.0, 1.0], [0.0, 1.0]),
    ]

    for epoch in range(2):
        print(f"\n{'=' * 70}")
        print(f"EPOCH {epoch + 1}")
        print(f"{'=' * 70}")

        for sample, target in training_data:
            pipeline.train_step([sample], [target])

    pipeline.summarize()

    print("\n" + "=" * 70)
    print("SYSTEM INTEROPERABILITY")
    print("=" * 70)

    try:
        import torch

        print("✓ PyTorch detected - DLPack interop available")
        st_tensor = st.Tensor(2, 2, [1.0, 2.0, 3.0, 4.0])
        print(f"  SpiralTorch tensor: {st_tensor.tolist()}")
        from torch.utils.dlpack import from_dlpack

        capsule = st_tensor.to_dlpack()
        pt_tensor = from_dlpack(capsule)
        print(f"  PyTorch tensor: {pt_tensor.tolist()}")
        pt_tensor *= 2
        print(f"  After PyTorch *= 2: {st_tensor.tolist()}")
        print("  ✓ Zero-copy memory sharing confirmed")
    except ImportError:
        print("  PyTorch not available - DLPack demo skipped")

    print("\n" + "=" * 70)
    print("INTEGRATION COMPLETE")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  1. Spiral consensus, robotics telemetry, and fractal vision align in feedback loops")
    print("  2. Drift safety oversight guides harmonized learning parameters")
    print("  3. Policy gradients adapt learning directly from system telemetry")
    print("  4. Geometry-aware monitoring keeps learning dynamics coherent")
    print("  5. SpiralTorch tensors interoperate seamlessly with PyTorch")
    print("\nSpiralTorch: A complete OS for Z-space machine learning")
    print("=" * 70 + "\n")


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
