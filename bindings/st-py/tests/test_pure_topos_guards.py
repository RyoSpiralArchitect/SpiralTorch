from __future__ import annotations

import pytest

st = pytest.importorskip("spiraltorch")


def test_zbox_python_contract_rejects_invalid_geometry_without_panicking() -> None:
    with pytest.raises(ValueError, match="zbox_density_non_positive"):
        st.ZBox([[0.0]], [1.0], density=0.0)

    zbox = st.ZBox([[0.0, 0.0]], [0.5], density=1.0)
    assert zbox.factor_dimension(0) == 2
    with pytest.raises(ValueError, match="zbox_factor_index"):
        zbox.factor_dimension(1)
    with pytest.raises(ValueError, match="zbox_curvature"):
        zbox.hyperbolic_volume(float("nan"))


def test_zbox_site_exposes_its_validated_radius_contract() -> None:
    site = st.ZBoxSite.default_for(-1.0).with_radius_window(0.1, 2.5)

    assert site.radius_min() == pytest.approx(0.1)
    assert site.radius_max() == pytest.approx(2.5)
    with pytest.raises(ValueError, match="zbox_radius_min"):
        site.with_radius_window(float("nan"), 2.5)
