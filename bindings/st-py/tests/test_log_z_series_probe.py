from __future__ import annotations

import pytest

st = pytest.importorskip("spiraltorch")


class _FakeLogZSeries:
    log_start = 0.0
    log_step = 0.25
    samples = [1.0, 2.0, 3.0, 4.0]
    weights = [0.0, 0.5, 0.5, 0.0]
    window = "hann"
    normalisation = "l1"

    def len(self) -> int:
        return len(self.samples)

    def evaluate_many_z(self, z_values):
        return [complex(1.0 + idx * 0.25, 0.1 * idx) for idx, _ in enumerate(z_values)]


def test_log_z_series_probe_exported_from_top_level_and_frac_module() -> None:
    assert "log_z_series_probe" in st.__all__
    assert "log_z_series_probe_to_zspace_partial" in st.__all__
    assert "log_z_series_partial" in st.__all__
    assert st.frac.log_z_series_probe is st.log_z_series_probe
    assert st.frac.log_z_series_probe_to_zspace_partial is st.log_z_series_probe_to_zspace_partial


def test_log_z_series_probe_matches_wasm_payload_shape() -> None:
    probe = st.log_z_series_probe(
        _FakeLogZSeries(),
        [0.5 + 0.0j, 0.2 + 0.3j],
        preview_len=2,
    )

    assert probe["kind"] == "spiraltorch.wasm_log_z_series_probe"
    assert probe["source_crate"] == "st-frac::cosmology"
    assert probe["mode"] == "log_z_series"
    assert probe["log_lattice"]["support"] == [0.0, 0.75]
    assert probe["options"] == {"window": "hann", "normalisation": "l1"}
    assert probe["sample_count"] == 4
    assert probe["sample_stats"]["energy"] == pytest.approx(7.5)
    assert probe["weight_stats"]["energy"] == pytest.approx(0.125)
    assert probe["z_count"] == 2
    assert probe["projection"]["preview_count"] == 2
    assert probe["projection"]["energy"] > 0.0


def test_log_z_series_probe_converts_to_zspace_partial() -> None:
    probe = st.log_z_series_probe(
        _FakeLogZSeries(),
        [0.5 + 0.0j, 0.2 + 0.3j],
        preview_len=2,
    )

    partial = st.log_z_series_probe_to_zspace_partial(
        probe,
        gradient_dim=6,
        telemetry_prefix="logz",
    )
    metrics = partial.resolved()
    telemetry = partial.telemetry_payload()

    assert isinstance(partial, st.ZSpacePartialBundle)
    assert partial.origin == "log_z_series:projection"
    assert 0.0 <= metrics["speed"] <= 1.0
    assert 0.0 <= metrics["memory"] <= 1.0
    assert 0.0 <= metrics["stability"] <= 1.0
    assert 0.0 <= metrics["drs"] <= 1.0
    assert len(metrics["gradient"]) == 6
    assert telemetry is not None
    assert telemetry["logz.sample_count"] == pytest.approx(4.0)
    assert telemetry["logz.z_count"] == pytest.approx(2.0)
    assert telemetry["logz.lattice.len"] == pytest.approx(4.0)


def test_log_z_series_partial_uses_native_series_when_available() -> None:
    if not callable(getattr(st.frac, "LogZSeries", None)):
        pytest.skip("native LogZSeries is unavailable")

    partial = st.log_z_series_partial(
        [1.0, 1.5, 2.0, 2.5],
        0.0,
        0.25,
        [0.5 + 0.0j, 0.2 + 0.3j],
        window="hann",
        gradient_dim=5,
    )
    metrics = partial.resolved()

    assert partial.origin == "log_z_series:projection"
    assert len(metrics["gradient"]) == 5
