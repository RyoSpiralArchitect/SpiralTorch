"""Public host-container regression tests; no pretrained model or speed claim."""

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
    assert model.forward(x).tolist() == [list(tape.output()[:3]), list(tape.output()[3:])]
    for factor in (0.3, -0.2, 0.0):
        seed = [factor] * 6
        dx, _ = tape.vjp(seed)
        actual = model.backward(x, st.Tensor(2, 3, seed)).tolist()
        assert actual == [list(dx[:3]), list(dx[3:])]
