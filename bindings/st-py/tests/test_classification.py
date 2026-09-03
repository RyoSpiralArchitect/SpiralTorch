"""Native classification contracts. No Python normalization or gradient fallback."""
import math

import pytest

import spiraltorch as st


pytestmark = pytest.mark.skipif(
    getattr(st, "_rs", None) is None,
    reason="Classification kernels require the native extension",
)


def tensor(rows):
    return st.Tensor(len(rows), len(rows[0]), [value for row in rows for value in row])


def flat(value):
    return [item for row in value.tolist() for item in row]


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
@pytest.mark.parametrize("smoothing", [0.0, 0.2, 1.0])
def test_all_python_entrypoints_share_the_native_kernel(reduction, smoothing):
    logits = tensor([[2.0, -1.0, 0.5], [0.2, 0.3, -0.4], [0.0, 0.0, 0.0]])
    labels = [0, 2, -100]
    options = {"reduction": reduction, "label_smoothing": smoothing}
    direct = logits.cross_entropy_with_logits(labels, **options)
    leaf = st.AutogradTensor.variable(logits)
    output = leaf.cross_entropy_with_logits(labels, **options)
    assert flat(output.value()) == flat(direct)
    assert output.shape() == ((3, 1) if reduction == "none" else (1, 1))
    seed = tensor([[1.0]] * output.shape()[0])
    expected = logits.cross_entropy_with_logits_backward(labels, seed, **options)
    assert flat(output.vector_jacobian_product(leaf, seed)) == flat(expected)
    assert leaf.grad() is None
    receipt = output.backward(seed)
    assert receipt["semantic_owner"] == "st-tensor"
    assert flat(leaf.grad()) == flat(expected)
    assert leaf.grad().is_snapshot()
    assert st.CrossEntropyWithLogits is st.nn.CrossEntropyWithLogits
    adapter = st.nn.CrossEntropyWithLogits(**options)
    target = tensor([[float(label)] for label in labels])
    assert flat(adapter(logits, target)) == flat(direct)
    assert flat(adapter.backward(logits, target)) == flat(expected)
    assert adapter.reduction == reduction
    assert adapter.ignore_index == -100
    assert adapter.label_smoothing == smoothing


def test_labels_and_logits_are_captured_before_python_mutation():
    source = tensor([[1.0, 2.0, -0.5], [0.2, -1.0, 0.3]])
    leaf = st.AutogradTensor.variable(source)
    labels = [0, 2]
    output = leaf.cross_entropy_with_logits(labels)
    expected = flat(source.cross_entropy_with_logits_backward(labels, tensor([[1.0]])))
    labels[:] = [-100, -100]
    source.add_row_inplace([4.0, -3.0, 2.0])
    output.backward()
    assert flat(leaf.grad()) == expected


def test_log_softmax_forward_and_vjp_are_shared():
    logits = tensor([[1.0, 2.0, -0.5], [0.2, -1.0, 0.3]])
    leaf = st.AutogradTensor.variable(logits)
    output = leaf.row_log_softmax()
    assert flat(output.value()) == flat(logits.row_log_softmax())
    for row in output.value().tolist():
        assert sum(math.exp(value) for value in row) == pytest.approx(1.0, abs=1e-7)
    seed = tensor([[0.3, -0.4, 0.7], [1.0, -1.0, 0.2]])
    output.backward(seed)
    assert flat(leaf.grad()) == flat(logits.row_log_softmax_backward(seed))


def test_tiny_tail_is_not_rounded_to_zero():
    logits = tensor([[0.0, -80.0]])
    loss = flat(logits.cross_entropy_with_logits([0]))[0]
    gradient = flat(logits.cross_entropy_with_logits_backward([0], tensor([[1.0]])))
    assert loss > 0.0
    assert loss == pytest.approx(math.exp(-80.0), rel=1e-6, abs=0.0)
    assert gradient == [-loss, loss]
    assert flat(logits.row_log_softmax()) == [-loss, -80.0]


@pytest.mark.parametrize("options", [
    {"reduction": "avg"}, {"label_smoothing": -0.1},
    {"label_smoothing": 1.1}, {"label_smoothing": math.nan},
    {"label_smoothing": math.inf},
])
def test_invalid_configuration_is_rejected_in_every_entrypoint(options):
    logits = tensor([[0.0, 1.0]])
    for value in [logits, st.AutogradTensor.variable(logits)]:
        with pytest.raises(ValueError):
            value.cross_entropy_with_logits([0], **options)
    with pytest.raises(ValueError):
        st.nn.CrossEntropyWithLogits(**options)


@pytest.mark.parametrize("labels", [[], [-1], [2], [2**63 - 1]])
def test_bad_labels_are_not_clamped(labels):
    with pytest.raises(ValueError):
        tensor([[0.0, 1.0]]).cross_entropy_with_logits(labels)


def test_fractional_class_indices_are_rejected_not_truncated():
    logits = tensor([[0.0, 1.0]])
    with pytest.raises(TypeError):
        logits.cross_entropy_with_logits([0.5])
    with pytest.raises(ValueError):
        st.nn.CrossEntropyWithLogits()(logits, tensor([[0.5]]))


def test_ignore_index_is_explicit_and_mean_cannot_hide_an_empty_training_batch():
    logits = tensor([[0.0, 1.0], [4.0, -4.0]])
    labels = [0, 99]
    loss = logits.cross_entropy_with_logits(labels, ignore_index=99)
    assert flat(loss) == flat(tensor([[0.0, 1.0]]).cross_entropy_with_logits([0]))
    with pytest.raises(ValueError, match="non-ignored labels"):
        logits.cross_entropy_with_logits([99, 99], ignore_index=99)
    assert flat(logits.cross_entropy_with_logits([99, 99], ignore_index=99, reduction="none")) == [0.0, 0.0]
    assert flat(logits.cross_entropy_with_logits([99, 99], ignore_index=99, reduction="sum")) == [0.0]
    bad = tensor([[0.0, 1.0], [math.nan, 0.0]])
    with pytest.raises(ValueError, match="non-finite"):
        bad.cross_entropy_with_logits(labels, ignore_index=99)


def test_unreduced_loss_requires_explicit_backward_seed():
    leaf = st.AutogradTensor.variable(tensor([[0.0, 1.0], [1.0, 0.0]]))
    output = leaf.cross_entropy_with_logits([1, 0], reduction="none")
    with pytest.raises(ValueError, match="explicit output gradient"):
        output.backward()
    output.backward(tensor([[1.0], [0.0]]))
    assert leaf.grad().tolist()[1] == [0.0, 0.0]


def test_existing_probability_loss_alias_is_unchanged():
    assert st.nn.SoftmaxCrossEntropy is st.nn.CategoricalCrossEntropy
    assert st.nn.CrossEntropy is st.nn.HyperbolicCrossEntropy
    assert st.nn.CrossEntropyWithLogits is not st.nn.SoftmaxCrossEntropy


def test_module_trainer_accepts_integer_target_loss_and_updates_parameters():
    trainer = st.nn.ModuleTrainer(backend="cpu", curvature=-1.0,
                                 hyper_learning_rate=0.01, fallback_learning_rate=0.01)
    model = st.nn.Sequential()
    model.add(st.nn.Linear("classification", 2, 3))
    trainer.prepare(model)
    before = {name: flat(value) for name, value in model.state_dict()}
    schedule = trainer.roundtable(3, 3, st.nn.RoundtableConfig(top_k=1, mid_k=1, bottom_k=1))
    inputs = tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    targets = tensor([[0.0], [1.0], [2.0]])
    stats = trainer.train_epoch(model, st.nn.CrossEntropyWithLogits(), [(inputs, targets)], schedule)
    assert stats.batches == 1
    assert any(flat(value) != before[name] for name, value in model.state_dict())
