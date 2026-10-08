import importlib.util
from pathlib import Path

import pytest
import torch
import spiraltorch as st

spec = importlib.util.spec_from_file_location("wave_gate_benchmark", Path(__file__).with_name("benchmark_wave_gate_learning.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


@pytest.mark.parametrize("radius", [None, -2., 0., 2.])
@pytest.mark.parametrize("origin", [False, True])
def test_matched_joint_vjps_and_finite_origin(radius, origin):
    rng = torch.Generator().manual_seed(19)
    values = [torch.randn(2, 7, 5, generator=rng), torch.randn(5, generator=rng),
              torch.randn(5, generator=rng) * .2]
    if origin:
        values[1].zero_()
        values[2].zero_()
    if radius is not None:
        values.append(torch.tensor(radius))
    dy = torch.randn(2, 7, 5, generator=rng)
    kernel = st.WaveGateKernel(curvature=-.7, porosity=.2)
    reference = bench.run_route("rust_list", values, dy, kernel)
    for name in bench.ROUTES:
        report = bench.check_result(bench.run_route(name, values, dy, kernel), reference,
                                    exact=name != "torch_reference")
        assert len(report) == len(values) + 1


def test_signed_zero_nonfinite_and_missing_gradient_are_not_accepted():
    with pytest.raises(ValueError, match="bits"):
        bench.check_result((torch.tensor(-0.),), (torch.tensor(0.),), exact=True)
    with pytest.raises(ValueError, match="nonfinite"):
        bench.check_result((torch.tensor(float("nan")),), (torch.tensor(0.),), exact=False)
    with pytest.raises(ValueError, match="arity"):
        bench.check_result((), (torch.tensor(0.),), exact=True)
