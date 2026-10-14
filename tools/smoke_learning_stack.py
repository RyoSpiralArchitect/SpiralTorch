"""Installed-wheel mechanics only: no model download, Torch, or GPU requirement."""

from array import array
import hashlib
import importlib
import importlib.machinery
import json
import math

import spiraltorch as st


def close(actual, expected, *, atol=3e-4, rtol=3e-3):
    assert len(actual) == len(expected), (len(actual), len(expected))
    for value, reference in zip(actual, expected):
        assert math.isfinite(value) and math.isfinite(reference), (value, reference)
        assert abs(value - reference) <= atol + rtol * abs(reference), (value, reference)


def finite_difference(function, values, *, epsilon=1e-3):
    result = []
    for index in range(len(values)):
        high, low = list(values), list(values)
        high[index] += epsilon
        low[index] -= epsilon
        result.append((function(high) - function(low)) / (2 * epsilon))
    return result


def dot(values, seed):
    assert len(values) == len(seed)
    return math.fsum(a * b for a, b in zip(values, seed))


def rejects_missing_forward(model, x, seed):
    try:
        model.backward(x, seed)
    except ValueError as error:
        assert "sequential_forward_missing" in str(error), error
    else:
        raise AssertionError("backward accepted an invalidated Sequential capture")


def sequential_capture():
    model = st.nn.Sequential()
    model.add(st.nn.Dropout(0.5, seed=47))
    x = st.Tensor(4, 32, [1.0] * 128)
    output = model.forward(x).tolist()
    assert {value for row in output for value in row} == {0.0, 2.0}
    model.zero_accumulators()
    assert model.backward(x, x).tolist() == output
    assert model.backward(x, x).tolist() == output
    model.eval()
    assert model.forward(x).tolist() == x.tolist()
    assert model.backward(x, x).tolist() == x.tolist()
    assert model.forward_untracked(x).tolist() == x.tolist()
    rejects_missing_forward(model, x, x)
    return "actual dropout mask retained; untracked capture rejected"


def topos_learning():
    kernel = st.ToposResonatorKernel(coupling=0.25, iterations=4, porosity=0.2)

    def learner():
        model = st.nn.Sequential()
        model.add_topos_resonator("gate", st.Tensor(1, 2, [0.0, 0.0]), kernel)
        return model

    model = learner()
    values, gate = [-0.7, 0.2, 0.4, 1.1], [0.3, -0.2]
    target = st.Tensor(2, 2, kernel.capture_shared_rows(values, gate, 2, 2).output)
    x = st.Tensor(2, 2, values)
    loss = st.nn.MeanSquaredError()
    initial = loss.forward(model.forward(x), target).tolist()[0][0]
    for _ in range(24):
        prediction = model.forward(x)
        seed = loss.backward(prediction, target)
        model.zero_accumulators()
        first = model.backward(x, seed).tolist()
        model.zero_accumulators()
        assert model.backward(x, seed).tolist() == first
        model.apply_step(0.1)
        rejects_missing_forward(model, x, seed)
    prediction = model.forward(x)
    final = loss.forward(prediction, target).tolist()[0][0]
    assert math.isfinite(initial) and math.isfinite(final) and 0 <= final < initial
    state = [(name, value.snapshot()) for name, value in model.state_dict()]
    assert any(value != 0 for _, tensor in state for row in tensor.tolist() for value in row)
    restored = learner()
    restored.load_state_dict(state)
    assert restored.forward(x).tolist() == prediction.tolist()
    return {"updates": 24, "initial_loss": initial, "final_loss": final,
            "handoff": "weights only; exact prediction, not optimizer resume"}


def wave_gate():
    kernel = st.WaveGateKernel(curvature=-0.7, saturation=1.0, porosity=0.2)
    x, gate, bias, seed = [0.2, -0.3, 0.6, 0.4], [0.5, -0.4], [0.1, -0.2], [0.3, -0.1, 0.4, 0.2]
    batch = kernel.forward(x, gate, bias, 2, 2)
    expected = batch.vjp(seed)
    for index, values in enumerate((x, gate, bias)):
        def objective(changed):
            inputs = [x, gate, bias]
            inputs[index] = changed
            return dot(kernel.forward(*inputs, 2, 2).output, seed)
        close(expected[index], finite_difference(objective, values))
    output = list(batch.output)
    x[:] = [float("nan")] * len(x)
    del kernel
    assert batch.output == output and batch.vjp(seed) == expected
    return "input/gate/bias VJPs checked; captured values owned"


def elliptic():
    warp = st.EllipticWarp(1.5, 4, 2)
    values = [0.3, 0.4, 0.8, 0.6, -0.3, -0.5]
    seed = [(index - 7) / 20 for index in range(18)]
    batch = warp.map_orientations_batch(values)
    gradient = batch.vjp(seed)
    close(gradient, finite_difference(
        lambda x: dot(warp.map_orientations_batch(x).features, seed), values))
    before = list(batch.features)
    warp.configure(spin_harmonics=5)
    assert batch.features == before and batch.vjp(seed) == gradient
    return "orientation VJP checked; capture survives recipe change"


def fractional_history():
    kernel = st.FractionalGlKernel(kernel_len=4)
    values = array("f", [0.2, -0.4, 0.8, 0.3, -0.1, 0.6, 0.7, -0.2])
    original = list(values)
    shape, alpha, log_gain = [1, 4, 2], 0.7, 0.2
    seed = [0.0] * 6 + [0.3, -0.4]
    batch = kernel.forward_history_log_gain_buffer(values, shape, 1, alpha, log_gain)
    dx, da, dg = batch.vjp(seed)
    assert batch.output[:2] == [0.0, 0.0] and dx[-2:] == [0.0, 0.0]
    assert da != 0 and dg != 0
    close(dx, finite_difference(lambda x: dot(
        kernel.forward_history_log_gain(x, shape, 1, alpha, log_gain).output, seed), original))
    close([da, dg], finite_difference(lambda p: dot(
        kernel.forward_history_log_gain(original, shape, 1, *p).output, seed), [alpha, log_gain]))
    assert batch.vjp_parameters(seed) == (da, dg)
    buffers = batch.vjp_buffer(array("f", seed))
    close(list(memoryview(buffers[0]).cast("f")), dx, atol=0, rtol=0)
    assert buffers[1:] == (da, dg)
    output = list(batch.output)
    values[:] = array("f", [float("nan")] * len(values))
    assert batch.output == output and batch.vjp(seed) == (dx, da, dg)
    return "causal input/order/gain VJPs checked; f32 buffer transport owned"


def resident_plan():
    model = st.nn.Sequential()
    model.add(st.nn.Linear(3, 3, name="in"))
    model.add(st.nn.Gelu())
    model.add(st.nn.LayerNorm("norm", 3, -1.0, 1e-5))
    model.add_topos_resonator("topos", st.Tensor(1, 3, [0.3, -0.2, 0.1]), st.ToposResonatorKernel())
    plan = model.inference_plan([2, 2, 3])
    payload = json.loads(plan.to_json())
    assert payload["schema"] == "spiraltorch.nn.inference_plan.v5"
    assert payload["input_shape"] == [2, 2, 3]
    assert [stage["kind"] for stage in payload["stages"]] == ["linear", "layer_norm", "topos_resonator"]
    assert st.nn.InferencePlan.from_json(plan.to_json()).to_json() == plan.to_json()
    return {"schema": payload["schema"], "gpu_execution": "not attempted"}


def byte_decoder_portable():
    """Exercise the installed native API without acquiring a GPU."""
    def check(condition, message):
        if not condition:
            raise AssertionError(message)

    shape = [1, 2, 2]
    def linear(cols, inner=2):
        return st.nn.InferencePlan.from_json(json.dumps({
            "schema": "spiraltorch.nn.inference_plan.v1",
            "input_shape": shape[:2] + [inner],
            "stages": [{"inner": inner, "cols": cols, "weight": [0.1] * (inner * cols),
                        "bias": [0.] * cols, "gelu": False}],
        }))
    one = linear(2)
    attention = st.nn.AttentionInferencePlan.from_projection_plans(
        one, one, one, one, heads=2, causal_offset=0)
    block = st.nn.ResidualAttentionPlan.from_plans(one, attention, one)
    plain = st.nn.ByteDecoderPlan.from_plans(
        st.Tensor(256, 2, [0.1] * 512), st.Tensor(2, 2, [0., 0.1] * 2), [block], linear(256))
    metrics = ["poincare_squared.v1", "euclidean_chord_squared.v1",
               "categorical_fisher_rao_squared.v1", "complex_projective_chord_squared.v1"]
    geometry = st.nn.ByteDecoderGeometryPlan(linear(2), [0.1], [0.2], [[0.3, -0.2]], -1.0)
    mixture = geometry.with_mixture(metrics, [[0.] * 8], [[1.] * 8])
    head_metrics = [[metrics[0], metrics[2]]]
    variants = [None] + [geometry.with_pair_metric(m) for m in metrics] + [
        geometry.with_head_pair_metrics(head_metrics), mixture,
        mixture.with_query_projections([linear(8, inner=2)])]
    for variant, version in zip(variants, [1, 1, 2, 2, 2, 3, 4, 5]):
        plan = plain if variant is None else plain.with_causal_geometry(variant)
        checkpoint = plan.initial_checkpoint()
        record = json.loads(checkpoint.to_json())
        check(record["schema"] == "spiraltorch.nn.byte_decoder_checkpoint.v" + str(version),
              "unexpected byte checkpoint schema")
        restored = st.nn.ByteDecoderCheckpoint.from_json(checkpoint.to_json())
        check(restored.to_json() == checkpoint.to_json(), "byte checkpoint changed")
        check(restored.attempted_revision == 0, "initial revision mismatch")
        check(restored.plan().parameter_layout() == plan.parameter_layout(), "byte layout changed")
        if version == 3:
            check(record["model"]["geometry"]["head_pair_metrics"] == head_metrics,
                  "heterogeneous routing was lost")
            check(restored.plan().causal_geometry.head_pair_metrics == head_metrics,
                  "restored heterogeneous routing was lost")
    data = st.nn.ByteLmBatch.from_documents([b"abc", b"xyz"], [(1, 0)], 2)
    check(data.shape == (1, 2) and data.input_bytes == b"xy" and data.target_bytes == b"yz",
          "byte selection/shift mismatch")
    try:
        st.nn.ByteLmBatch.from_documents([b"abc"], [(0, 1)], 2)
    except ValueError:
        pass
    else:
        raise AssertionError("cross-document window was accepted")
    return {"modes": len(variants), "checkpoint_roundtrip": "exact", "gpu_execution": "not attempted"}


BYTE_CORPUS_EXPORTS = (
    "ByteCorpusStudy", "ByteCorpusStudyCheckpoint", "ByteCorpusStudySegment",
    "ByteCorpusPreparedBiasStudy", "ByteCorpusBiasCalibration",
    "ByteCorpusHeadwiseControls", "ByteCorpusMixtureControls", "ByteCorpusQueryMixtureControls",
)


def byte_corpus_portable():
    """Check packaged study contracts, not training, without repository fixtures."""
    native = importlib.import_module("spiraltorch.spiraltorch")
    # Attribute forwarding can populate __all__; inspect before any lazy lookup.
    public_exports, native_exports = set(st.nn.__all__), set(native.nn.__all__)
    for name in BYTE_CORPUS_EXPORTS:
        assert name in public_exports and name in native_exports, name
        assert getattr(st.nn, name) is getattr(native.nn, name), name
    limit = "BYTE_CORPUS_STUDY_MAX_BYTES"
    assert limit in public_exports and limit in native_exports
    assert getattr(st.nn, limit) == getattr(native.nn, limit) == 64 * 1024 * 1024

    def parameter(name, shape):
        return {"name": name, "shape": shape, "values": [0.1] * math.prod(shape)}

    # Deliberately fabricated constant weights: this only exercises admission.
    backbone = [parameter("token_embedding", [256, 2]),
                parameter("position_embedding", [2, 2])]
    for prefix, first, second in (
        ("block.0.pre", [2], [2]), ("block.0.qkv", [2, 6], [6]),
        ("block.0.output", [2, 2], [2]), ("block.0.feed", [2], [2]),
        ("block.0.up", [2, 2], [2]), ("block.0.down", [2, 2], [2]),
        ("head", [2], [2]), ("head.output", [2, 256], [256]),
    ):
        backbone += [parameter(prefix + (".gain" if len(first) == 1 else ".weight"), first),
                     parameter(prefix + ".bias", second)]
    geometry = [parameter(name, shape) for name, shape in (
        ("geometry.projection.weight", [2, 2]), ("geometry.projection.bias", [2]),
        ("geometry.raw_decay", [1]), ("geometry.raw_phase", [1]),
        ("geometry.raw_gain.0", [1]),
    )]
    request = {
        "schema": "spiraltorch.byte_corpus.request.v1",
        "config": dict(batch=1, steps=2, width=2, hidden=2, heads=1, blocks=[False],
                       geometry_cols=2, curvature=-0.75),
        "cases": [
            dict(name="plain", seed=11, geometry=False, parameters=backbone),
            dict(name="geometry", seed=11, geometry=True,
                 parameters=backbone[:2] + geometry + backbone[2:]),
        ],
        "train_documents": [[1, 2, 3, 4, 5]], "validation_documents": [[8, 9, 10]],
        "train_batches": [[[0, 0]], [[0, 2]]], "validation_batches": [[[0, 0]]],
        "checkpoint_every": 2, "rate": 0.05,
    }
    raw = json.dumps(request, separators=(",", ":"), allow_nan=False).encode("utf-8")
    study = st.nn.ByteCorpusStudy.from_json(raw)
    assert study.request_sha256 == hashlib.sha256(raw).hexdigest()
    assert st.nn.ByteCorpusStudy.from_json(raw.decode("utf-8")).request_sha256 == study.request_sha256
    assert st.nn.ByteCorpusStudy.from_json(raw + b" ").request_sha256 != study.request_sha256
    assert study.total_updates == 2 and 0 < study.checkpoint_size_bound() <= getattr(st.nn, limit)
    study.validate_segment(0)
    study.validate_segment()
    for error, call in (
        (TypeError, lambda: st.nn.ByteCorpusStudy.from_json(request)),
        (TypeError, lambda: study.validate_segment(True)),
        (ValueError, lambda: study.validate_segment(3)),
        (ValueError, lambda: study.checkpoint_from_json("{}")),
    ):
        try:
            call()
        except error:
            pass
        else:
            raise AssertionError("invalid study input was accepted")
    return {"native_exports": len(BYTE_CORPUS_EXPORTS), "request_identity": "exact bytes",
            "segment_validation": "CPU-only", "gpu_execution": "not attempted"}


def main():
    native = importlib.import_module("spiraltorch.spiraltorch")
    assert any(str(native.__file__).endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES)
    for name in ("ToposResonatorKernel", "WaveGateKernel", "FractionalGlKernel", "EllipticWarp"):
        assert getattr(st, name) is getattr(native, name), name
    report = {
        "scope": "installed native mechanics; no GPU, pretrained FT, quality, or speed claim",
        "sequential": sequential_capture(),
        "topos": topos_learning(),
        "wave_gate": wave_gate(),
        "elliptic": elliptic(),
        "fractional_history": fractional_history(),
        "resident_plan": resident_plan(),
        "byte_decoder": byte_decoder_portable(),
        "byte_corpus": byte_corpus_portable(),
    }
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return report


if __name__ == "__main__":
    main()
