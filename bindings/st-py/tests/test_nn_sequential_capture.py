"""Public host-container regression tests; no pretrained model or speed claim."""

import json
from pathlib import Path

import pytest
import spiraltorch as st


def test_dropout_backward_reuses_the_forward_mask():
    model = st.nn.Sequential()
    model.add(st.nn.Dropout(0.5, seed=47))
    x = st.Tensor(4, 32, [1.0] * 128)
    output = model.forward(x).tolist()
    assert model.backward(x, x).tolist() == output
    assert model.backward(x, x).tolist() == output


def test_eval_preserves_gradients_but_untracked_invalidates_container_tape():
    model = st.nn.Sequential()
    model.add(st.nn.Dropout(0.5, seed=47))
    x = st.Tensor(1, 128, [1.0] * 128)
    model.eval()
    assert model.forward(x).tolist() == x.tolist()
    assert model.backward(x, x).tolist() == x.tolist()
    assert model.forward_untracked(x).tolist() == x.tolist()
    with pytest.raises(ValueError, match="sequential_forward_missing"):
        model.backward(x, x)
    model.train()
    assert model.forward_untracked(x).tolist() != x.tolist()


def test_input_mismatch_rejects_before_pullback_and_allows_retry():
    model = st.nn.Sequential()
    model.add(st.nn.Gelu())
    x = st.Tensor(1, 2, [0.25, -0.5])
    model.forward(x)
    with pytest.raises(ValueError, match="sequential_forward_input_mismatch"):
        model.backward(st.Tensor(1, 2, [0.25, 0.5]), x)
    assert model.backward(x, x).shape() == x.shape()


def test_topos_sequence_matches_direct_rust_kernel_on_repeated_pullbacks():
    kernel = st.ToposResonatorKernel(coupling=0.25, iterations=5, porosity=0.3,
                                    saturation=1.0, max_values=6)
    gate = [0.8, -0.4, 1.1]
    values = [-2.0, 0.4, 2.0, 0.2, -1.5, 3.0]
    model = st.nn.Sequential()
    model.add_topos_resonator("topos", st.Tensor(1, 3, gate), kernel)
    x = st.Tensor(2, 3, values)
    tape = kernel.capture_shared_rows(values, gate, 2, 3)
    assert model.forward(x).tolist() == [list(tape.output[:3]), list(tape.output[3:])]
    for factor in (0.3, -0.2, 0.0):
        model.zero_accumulators()
        seed = [factor] * 6
        dx, _ = tape.vjp(seed)
        actual = model.backward(x, st.Tensor(2, 3, seed)).tolist()
        assert actual == [list(dx[:3]), list(dx[3:])]


@pytest.mark.parametrize("with_topos", [False, True])
@pytest.mark.parametrize("seed", [47, 91])
def test_multistep_learning_matches_independent_torch(monkeypatch, with_topos, seed):
    torch = pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    from benchmark_topos_learning import torch_reference

    model = st.nn.Sequential()
    model.add(st.nn.Dropout(0.5, seed=seed))
    model.add(st.nn.Linear("first", 3, 3))
    kernel = st.ToposResonatorKernel(coupling=0.25, iterations=5, porosity=0.3,
                                    saturation=1.0, max_values=12)
    initial = {
        "first::weight": [[0.25, -0.4, 0.6], [0.1, 0.5, -0.2], [-0.3, 0.2, 0.7]],
        "first::bias": [[0.2, -0.1, 0.05]],
        "last::weight": [[0.4, -0.2], [0.1, 0.3], [-0.5, 0.2]],
        "last::bias": [[0.15, -0.05]],
    }
    if with_topos:
        initial["topos"] = [[0.8, -0.4, 1.1]]
        model.add_topos_resonator("topos", st.Tensor(1, 3, initial["topos"][0]), kernel)
    model.add(st.nn.Gelu())
    model.add(st.nn.Linear("last", 3, 2))

    def native(value):
        return st.Tensor(value.shape[0], value.shape[1], value.detach().flatten().tolist())

    parameters = {
        name: torch.tensor(values, dtype=torch.float32, device="cpu", requires_grad=True)
        for name, values in initial.items()
    }
    model.load_state_dict([(name, native(value)) for name, value in parameters.items()])
    optimizer = torch.optim.SGD(parameters.values(), lr=0.03125)
    loss = st.nn.MeanSquaredError()
    config = json.loads(kernel.configuration_json())
    maximum = dict(output=0., loss=0., input_gradient=0., parameter=0.)

    def compare(name, actual, expected):
        actual = torch.tensor(actual.tolist(), dtype=torch.float32, device="cpu")
        expected = expected.detach().reshape(actual.shape)
        torch.testing.assert_close(actual, expected, rtol=5e-4, atol=3e-5)
        maximum[name] = max(maximum[name], float((actual - expected).abs().max()))

    # Match masks, not unrelated RNG implementations. This separate stream is
    # advanced only by forward, so a container replay also breaks later steps.
    masks = st.nn.Dropout(0.5, seed=seed)
    ones = st.Tensor(4, 3, [1.] * 12)
    observed_masks = set()
    for step in range(12):
        values = [((i * 17 + step * 7) % 31) / 7. - 2. for i in range(12)]
        x = torch.tensor(values, dtype=torch.float32, device="cpu").reshape(4, 3).requires_grad_()
        target = torch.tensor([((i + step) % 7) / 10. - 0.3 for i in range(8)],
                              dtype=torch.float32, device="cpu").reshape(4, 2)
        mask = torch.tensor(masks.forward(ones).tolist(), dtype=torch.float32, device="cpu")
        assert set(mask.flatten().tolist()) == {0., 2.}
        observed_masks.add(tuple(mask.flatten().tolist()))
        optimizer.zero_grad()
        hidden = (x * mask) @ parameters["first::weight"] + parameters["first::bias"]
        if with_topos:
            hidden = torch_reference(hidden, parameters["topos"], config)
        expected = (torch.nn.functional.gelu(hidden, approximate="tanh")
                    @ parameters["last::weight"] + parameters["last::bias"])
        expected_loss = (expected - target).square().mean()
        native_x, native_target = native(x), native(target)
        output = model.forward(native_x)
        compare("output", output, expected)
        compare("loss", loss.forward(output, native_target), expected_loss)
        cotangent = loss.backward(output, native_target)
        model.zero_accumulators()
        for factor in (0.25, 0.75):
            dx = torch.autograd.grad(expected_loss * factor, x, retain_graph=True)[0]
            compare("input_gradient", model.backward(native_x, cotangent.scale(factor)), dx)
        expected_loss.backward()
        optimizer.step()
        model.apply_step(0.03125)
        actual_parameters = dict(model.state_dict())
        assert actual_parameters.keys() == parameters.keys()
        for name, expected_parameter in parameters.items():
            compare("parameter", actual_parameters[name], expected_parameter)
        with pytest.raises(ValueError, match="sequential_forward_missing"):
            model.backward(native_x, cotangent)
    assert len(observed_masks) > 1
    for name, parameter in parameters.items():
        assert not torch.equal(parameter.detach(), torch.tensor(initial[name], dtype=torch.float32))
    print(json.dumps(dict(scope="12 synthetic CPU Torch-matched updates; no speed or quality claim",
                          seed=seed, with_topos=with_topos, max_abs_error=maximum)))
