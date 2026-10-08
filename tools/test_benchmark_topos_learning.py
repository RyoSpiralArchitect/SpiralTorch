import importlib.util
import json
from pathlib import Path

import pytest
import torch
import spiraltorch as st

spec = importlib.util.spec_from_file_location("topos_benchmark", Path(__file__).with_name("benchmark_topos_learning.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


@pytest.mark.parametrize("iterations,coupling", [(1, 0.), (4, .25), (16, .75)])
@pytest.mark.parametrize("porosity", [0., .2])
@pytest.mark.parametrize("origin", [False, True])
def test_matched_finite_unroll_and_both_vjps(iterations, coupling, porosity, origin):
    rng = torch.Generator().manual_seed(19)
    values = [torch.randn(2, 7, 5, generator=rng), torch.randn(5, generator=rng)]
    if origin:
        values[1].zero_()
    dy = torch.randn(2, 7, 5, generator=rng)
    kernel = st.ToposResonatorKernel(coupling=coupling, iterations=iterations, porosity=porosity)
    config = json.loads(kernel.configuration_json())
    names = bench.routes(kernel, include_expanded_capture=True)
    references = bench.route_references(values, dy, kernel, config, names)
    for name in names:
        assert len(bench.check_result(bench.run_route(name, values, dy, kernel, config), references[name],
                                     exact=name != "torch_reference")) == 3
    bench.check_result(bench.run_route("rust_public", values, dy, kernel, config),
                       references["rust_list"], exact=False)


def test_shared_reference_keeps_wide_cancellation_and_rejects_wrong_gate_bits():
    kernel = st.ToposResonatorKernel(coupling=0., iterations=1)
    values = [torch.tensor([[[2.**24], [1.], [-2.**24]]]), torch.zeros(1)]
    dy = torch.ones_like(values[0])
    config = json.loads(kernel.configuration_json())
    references = bench.route_references(values, dy, kernel, config, bench.routes(kernel))
    assert references["rust_public"][2].item() == 1.
    actual = bench.run_route("rust_public", values, dy, kernel, config)
    bench.check_result(actual, references["rust_public"], exact=True)
    for wrong in (torch.zeros(1), torch.tensor([1. / 3.])):
        with pytest.raises(ValueError, match="bits"):
            bench.check_result((*actual[:2], wrong), references["rust_public"], exact=True)


def test_signed_zero_nonfinite_missing_gradient_and_unbalanced_order_rejected():
    zero = (torch.tensor(0.),) * 3
    for actual, label in [((torch.tensor(-0.), *zero[1:]), "bits"),
                          ((torch.tensor(float("nan")), *zero[1:]), "nonfinite"), ((), "arity")]:
        with pytest.raises(ValueError, match=label):
            bench.check_result(actual, zero, exact=True)
    for count in (3, 4, 5):
        names = list(range(count))
        orders = bench.round_orders(names, count * 3)
        for position in range(count):
            assert all(sum(row[position] == n for row in orders) == 3 for n in names)
        with pytest.raises(ValueError, match="balance"):
            bench.round_orders(names, 1)
