from __future__ import annotations

from unittest.mock import patch

from spiral.hypergrad import (
    hypergrad_session,
    hypergrad_summary_dict,
    suggest_hypergrad_operator,
)


class _FakeSummary:
    def __init__(
        self,
        *,
        l1: float = 1.0,
        l2: float = 0.5,
        linf: float = 0.8,
        mean_abs: float = 0.3,
        rms: float = 0.4,
        count: int = 12,
        sum_squares: float = 0.2,
        total: float = 0.1,
        sum_cubes: float = 0.05,
        sum_quartic: float = 0.025,
        mean: float = 0.02,
        variance: float = 0.015,
        std: float = 0.122474487,
        skewness: float = 0.1,
        kurtosis: float = 3.0,
        min_value: float = -0.4,
        max_value: float = 0.9,
        support_width: float = 1.3,
        positive_count: int = 7,
        negative_count: int = 5,
        zero_count: int = 0,
        near_zero_count: int = 1,
        positive_fraction: float = 0.6,
        negative_fraction: float = 0.4,
        zero_fraction: float = 0.0,
        near_zero_fraction: float = 0.05,
        activation: float = 0.95,
        sign_lean: float = 0.2,
        sign_entropy: float = 0.85,
    ) -> None:
        self._values = {
            "l1": float(l1),
            "l2": float(l2),
            "linf": float(linf),
            "mean_abs": float(mean_abs),
            "rms": float(rms),
            "count": int(count),
            "sum_squares": float(sum_squares),
            "sum": float(total),
            "sum_cubes": float(sum_cubes),
            "sum_quartic": float(sum_quartic),
            "mean": float(mean),
            "variance": float(variance),
            "std": float(std),
            "skewness": float(skewness),
            "kurtosis": float(kurtosis),
            "min": float(min_value),
            "max": float(max_value),
            "support_width": float(support_width),
            "positive_count": int(positive_count),
            "negative_count": int(negative_count),
            "zero_count": int(zero_count),
            "near_zero_count": int(near_zero_count),
            "positive_fraction": float(positive_fraction),
            "negative_fraction": float(negative_fraction),
            "zero_fraction": float(zero_fraction),
            "near_zero_fraction": float(near_zero_fraction),
            "activation": float(activation),
            "sign_lean": float(sign_lean),
            "sign_entropy": float(sign_entropy),
        }

    def l1(self) -> float:
        return self._values["l1"]

    def l2(self) -> float:
        return self._values["l2"]

    def linf(self) -> float:
        return self._values["linf"]

    def mean_abs(self) -> float:
        return self._values["mean_abs"]

    def rms(self) -> float:
        return self._values["rms"]

    def count(self) -> int:
        return self._values["count"]

    def sum_squares(self) -> float:
        return self._values["sum_squares"]

    def sum(self) -> float:
        return self._values["sum"]

    def sum_cubes(self) -> float:
        return self._values["sum_cubes"]

    def sum_quartic(self) -> float:
        return self._values["sum_quartic"]

    def mean(self) -> float:
        return self._values["mean"]

    def variance(self) -> float:
        return self._values["variance"]

    def std(self) -> float:
        return self._values["std"]

    def skewness(self) -> float:
        return self._values["skewness"]

    def kurtosis(self) -> float:
        return self._values["kurtosis"]

    def min(self) -> float:
        return self._values["min"]

    def max(self) -> float:
        return self._values["max"]

    def support_width(self) -> float:
        return self._values["support_width"]

    def positive_count(self) -> int:
        return self._values["positive_count"]

    def negative_count(self) -> int:
        return self._values["negative_count"]

    def zero_count(self) -> int:
        return self._values["zero_count"]

    def near_zero_count(self) -> int:
        return self._values["near_zero_count"]

    def positive_fraction(self) -> float:
        return self._values["positive_fraction"]

    def negative_fraction(self) -> float:
        return self._values["negative_fraction"]

    def zero_fraction(self) -> float:
        return self._values["zero_fraction"]

    def near_zero_fraction(self) -> float:
        return self._values["near_zero_fraction"]

    def activation(self) -> float:
        return self._values["activation"]

    def sign_lean(self) -> float:
        return self._values["sign_lean"]

    def sign_entropy(self) -> float:
        return self._values["sign_entropy"]


class _FakeHypergrad:
    def __init__(self) -> None:
        self._reset_calls = 0
        self._summary = _FakeSummary()
        self._gradient = [0.1, -0.2, 0.05]

    def curvature(self) -> float:
        return -1.0

    def learning_rate(self) -> float:
        return 0.05

    def shape(self) -> tuple[int, int]:
        return (2, 2)

    def summary(self) -> _FakeSummary:
        return self._summary

    def gradient(self) -> list[float]:
        return list(self._gradient)

    def reset(self) -> None:
        self._reset_calls += 1

    @property
    def reset_calls(self) -> int:
        return self._reset_calls


def test_hypergrad_session_resets_and_invokes_callback() -> None:
    fake = _FakeHypergrad()
    applied: list[list[float]] = []

    def _apply(tape: _FakeHypergrad) -> None:
        applied.append(tape.gradient())

    with patch("spiral.hypergrad.st.hypergrad", return_value=fake) as factory:
        with hypergrad_session(2, 2, apply=_apply) as tape:
            assert tape is fake
        factory.assert_called_once_with(2, 2, curvature=-1.0, learning_rate=0.05, topos=None)

    assert fake.reset_calls == 1
    assert applied == [fake.gradient()]


def test_hypergrad_summary_dict_includes_gradient_and_extra_metrics() -> None:
    fake = _FakeHypergrad()
    metrics = hypergrad_summary_dict(fake, include_gradient=True, extra={"hypergrad_norm": 0.42})

    assert metrics["shape"] == (2, 2)
    assert metrics["curvature"] == -1.0
    assert metrics["learning_rate"] == 0.05
    assert metrics["summary"]["hypergrad_norm"] == 0.42
    assert metrics["gradient"] == fake.gradient()
    assert "std" in metrics["summary"]
    assert "skewness" in metrics["summary"]
    assert "activation" in metrics["summary"]
    assert metrics["summary"]["positive_count"] == fake.summary().positive_count()


def test_suggest_hypergrad_operator_clamps_when_requested() -> None:
    payload = {
        "summary": {
            "mean_abs": 10.0,
            "rms": 0.01,
            "l2": 0.001,
            "linf": 20.0,
            "count": 16,
        }
    }

    hints = suggest_hypergrad_operator(payload, clamp=True)
    assert hints["mix"] == 0.9
    assert hints["gain"] == 3.0
    assert hints["count"] == 16.0
    assert hints["spread"] > 0.0
    assert "std" in hints
    assert "kurtosis" in hints
    assert "activation" in hints
    assert "support_width" in hints


def test_suggest_hypergrad_operator_accepts_tape_instances() -> None:
    fake = _FakeHypergrad()
    hints = suggest_hypergrad_operator(fake, clamp=False)

    assert 0.0 < hints["ratio"]
    assert 0.0 < hints["mix"]
    assert abs(hints["mix"] - hints["ratio"]) < max(1e-6, hints["ratio"] * 0.2)
    assert hints["count"] == float(fake.summary().count())
    assert "skewness" in hints
    assert "sign_entropy" in hints
