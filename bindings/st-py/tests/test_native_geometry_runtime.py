"""Installed-wheel learning and artifact continuation without optional ML packages."""

from array import array
import copy
import importlib.abc
import json
from pathlib import Path
import sys
import unittest


class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "transformers", "accelerate"}:
            raise AssertionError("native geometry runtime must not import " + fullname)


if __name__ == "__main__":
    assert not any(name.split(".")[0] in {"torch", "transformers", "accelerate"} for name in sys.modules)
    sys.meta_path.insert(0, NoTorch())

import spiraltorch as st  # noqa: E402 - install the forbidden-import guard first.

FIXTURE = json.loads((Path(__file__).resolve().parents[3] /
                     "crates/st-nn/tests/fixtures/channel_affine.fixture.json").read_text())


def artifact(kind="affine", *, frozen_gate=False):
    state = copy.deepcopy(FIXTURE["artifact"])
    p = state["placements"][0]
    p["path"] = "site"
    if kind in {"elliptic", "anchored", "causal", "gated"}:
        p["recipe"] = {
            "schema": {"elliptic": "spiraltorch.elliptic_residual_adapter.v1",
                       "anchored": "spiraltorch.elliptic_anchored_residual_adapter.v1",
                       "causal": "spiraltorch.elliptic_causal_residual_adapter.v1",
                       "gated": "spiraltorch.elliptic_gated_causal_residual_adapter.v1"}[kind],
            "features": 3, "strength": .5,
            "warp": {"curvature_radius": 1.3, "sheet_count": 3, "spin_harmonics": 2},
        }
        p["tensors"] = [
            {"name": "orientation.weight", "shape": [2, 3], "values": [.2, -.1, .15, .05, .3, -.2], "requires_grad": True, "role": "parameter"},
            {"name": "orientation.bias", "shape": [2], "values": [.1, -.2], "requires_grad": True, "role": "parameter"},
            {"name": "readout.weight", "shape": [3, 9], "values": [(i - 13) * .01 for i in range(27)], "requires_grad": True, "role": "parameter"},
        ]
        if kind in {"anchored", "gated"}:
            p["tensors"].append({"name": "raw_mix", "shape": [], "values": [.3], "requires_grad": True, "role": "parameter"})
    elif kind != "affine":
        p["recipe"] = {"schema": "spiraltorch.wave_gate_adapter.v1" if kind == "wave" else "spiraltorch.wave_gate_adapter.v2",
                       "features": 3, "strength": .5,
                       "kernel": {"curvature": -1., "saturation": 1., "porosity": .05, "max_values": 64}}
        if kind != "wave":
            p["recipe"]["learnable_radius"] = kind == "learned"
            p["tensors"].append({"name": "log_radius", "shape": [], "values": [0.],
                                 "requires_grad": kind == "learned", "role": "parameter" if kind == "learned" else "buffer"})
    if frozen_gate:
        next(t for t in p["tensors"] if t["name"] == "gate")["requires_grad"] = False
    return st.GeometryAdapterArtifact.create_json(json.dumps(state))


class NativeGeometryTests(unittest.TestCase):
    def test_shared_fixture_residual_vjp_and_owned_buffers(self):
        f = FIXTURE
        runtime = st.NativeGeometryRuntime(artifact())
        values = array("f", f["input"])
        batch = runtime.forward_buffer("site", values, f["rows"])
        expected = [x + .5 * y for x, y in zip(f["input"], f["output"])]
        self.assertEqual(batch.output, expected)
        self.assertEqual(batch.output_buffer(), array("f", expected).tobytes())
        self.assertEqual(batch.shape, (2, 3))
        self.assertEqual(batch.source_artifact_id, runtime.artifact().artifact_id)
        values[0] = 999.
        exported = batch.output_buffer()
        exported[0] ^= 255
        self.assertEqual(batch.output, expected)
        gx, parameters = batch.vjp(f["upstream"])
        self.assertEqual(gx, [a + .5 * b for a, b in zip(f["upstream"], f["grad_input"])])
        self.assertEqual(parameters, {"gate": [.5 * x for x in f["grad_gate"]], "bias": [.5 * x for x in f["grad_bias"]]})
        buffers = batch.vjp_buffer(array("f", f["upstream"]))
        self.assertEqual(buffers[0], array("f", gx).tobytes())
        self.assertEqual(buffers[1], {name: array("f", x).tobytes() for name, x in parameters.items()})
        self.assertEqual(runtime.execution_backend, "rust_f32_cpu")
        self.assertIn("NativeGeometryRuntime", st.__all__)
        self.assertIn("NativeGeometryBatch", st.__all__)

    def test_native_learning_and_exact_save_reload_continuation(self):
        x, y = array("f", [.2, -.3, .4, .5, .1, -.2]), array("f", [.1] * 6)
        for kind in ("affine", "wave", "fixed", "learned", "elliptic", "anchored"):
            with self.subTest(kind=kind):
                runtime = st.NativeGeometryRuntime(artifact(kind))
                first = runtime.train_mse_buffer("site", x, y, 2, .2)
                for _ in range(39):
                    runtime.train_mse_buffer("site", x, y, 2, .2)
                saved = runtime.artifact()
                loaded = st.GeometryAdapterArtifact.from_json(saved.to_json(), saved.artifact_id, saved.base_model_identity)
                restored = st.NativeGeometryRuntime(loaded)
                self.assertEqual(runtime.forward_buffer("site", x, 2).output, restored.forward("site", list(x), 2).output)
                loss = runtime.train_mse("site", list(x), list(y), 2, .2)
                self.assertLess(loss, first)
                self.assertEqual(loss, restored.train_mse_buffer("site", x, y, 2, .2))
                self.assertEqual(runtime.artifact().artifact_id, restored.artifact().artifact_id)

    def test_fixed_and_frozen_parameters_do_not_update(self):
        original = artifact("fixed", frozen_gate=True)
        runtime = st.NativeGeometryRuntime(original)
        batch = runtime.forward("site", [.2] * 6, 2)
        self.assertNotIn("log_radius", batch.vjp([1.] * 6)[1])
        runtime.step_sgd(batch, [1.] * 6, .1)
        before = {t["name"]: t for t in json.loads(original.state_json())["placements"][0]["tensors"]}
        after = {t["name"]: t for t in json.loads(runtime.artifact().state_json())["placements"][0]["tensors"]}
        self.assertEqual(before["log_radius"], after["log_radius"])
        self.assertEqual(before["gate"], after["gate"])
        self.assertNotEqual(before["bias"], after["bias"])
        updated = runtime.artifact().artifact_id
        with self.assertRaisesRegex(ValueError, "stale"):
            runtime.step_sgd(batch, [1.] * 6, .1)
        self.assertEqual(runtime.artifact().artifact_id, updated)

    def test_elliptic_owned_composition_frozen_projection_and_row_budget(self):
        class MustNotRead:
            def __len__(self): raise AssertionError("reject row budget before input access")
        for kind in ("elliptic", "anchored"):
            state = json.loads(artifact(kind).state_json())
            for tensor in state["placements"][0]["tensors"]:
                if tensor["name"].startswith("orientation."):
                    tensor["requires_grad"] = False
            saved = st.GeometryAdapterArtifact.create_json(json.dumps(state))
            runtime = st.NativeGeometryRuntime(saved)
            values = array("f", [.2, -.3, .4, .5, .1, -.2])
            batch = runtime.forward_buffer("site", values, 2)
            output, pullback = batch.output, batch.vjp([1.] * 6)
            values[0] = float("nan")
            exported = batch.output_buffer()
            exported[0] ^= 255
            runtime.step_sgd(batch, [1.] * 6, .01)
            self.assertEqual(batch.output, output)
            self.assertEqual(batch.vjp([1.] * 6), pullback)
            before = {t["name"]: t for t in state["placements"][0]["tensors"]}
            after = {t["name"]: t for t in json.loads(runtime.artifact().state_json())["placements"][0]["tensors"]}
            for name in ("orientation.weight", "orientation.bias"):
                self.assertEqual(before[name], after[name])
            self.assertNotEqual(before["readout.weight"], after["readout.weight"])
            with self.assertRaises(ValueError):
                runtime.forward("site", MustNotRead(), 65_537)
            empty = runtime.forward("site", [], 0)
            self.assertEqual(empty.output, [])
            self.assertTrue(all(x == 0. for v in empty.vjp([])[1].values() for x in v))

    def test_elliptic_causal_recipe_is_not_silently_flattened(self):
        for kind in ("causal", "gated"):
            runtime = st.NativeGeometryRuntime(artifact(kind))
            with self.assertRaisesRegex(ValueError, "currently supports"):
                runtime.forward("site", [.2] * 6, 2)
            result = runtime.forward_sequence("site", [.2] * 6, (1, 2))
            self.assertEqual(result.sequence_shape, (1, 2, 3))
            self.assertEqual(result.shape, (2, 3))

    def test_native_sequence_learning_and_saved_context_continuation(self):
        x = array("f", [(i * 7 % 19) * .05 - .3 for i in range(18)])
        y = array("f", [.1] * 18)
        for kind in ("causal", "gated"):
            runtime = st.NativeGeometryRuntime(artifact(kind))
            snapshot = runtime.forward_sequence_buffer("site", x, (2, 3))
            first = runtime.train_sequence_mse_buffer("site", x, y, (2, 3), .2)
            for _ in range(39):
                runtime.train_sequence_mse("site", list(x), list(y), (2, 3), .2)
            saved = runtime.artifact()
            restored = st.NativeGeometryRuntime(st.GeometryAdapterArtifact.from_json(
                saved.to_json(), saved.artifact_id, saved.base_model_identity))
            loss = runtime.train_sequence_mse_buffer("site", x, y, (2, 3), .2)
            self.assertLess(loss, first)
            self.assertEqual(loss, restored.train_sequence_mse("site", list(x), list(y), (2, 3), .2))
            self.assertEqual(runtime.artifact().artifact_id, restored.artifact().artifact_id)
            before = runtime.artifact().artifact_id
            with self.assertRaisesRegex(ValueError, "stale"):
                runtime.step_sgd(snapshot, [1.] * 18, .1)
            with self.assertRaises(ValueError):
                runtime.train_sequence_mse("site", list(x), [float("nan")] * 18, (2, 3), .1)
            self.assertEqual(runtime.artifact().artifact_id, before)

    def test_sequence_admission_is_before_foreign_values_and_preserves_pointwise_route(self):
        class MustNotRead:
            def __len__(self): raise AssertionError("shape admission must precede values")
        class OversizedTarget:
            def __len__(self): return 10**10
            def __getitem__(self, _): raise AssertionError("target length must precede extraction")
        runtime = st.NativeGeometryRuntime(artifact("gated"))
        invalid = ((), (1, 2, 3), (True, 3), (-1, 3), (1., 3), (0, 1), (1, 0),
                   (2**32, 2), (65_537, 1), (1, 1025))
        for shape in invalid:
            for method in (runtime.forward_sequence, runtime.forward_sequence_buffer):
                with self.assertRaises((TypeError, ValueError, OverflowError)):
                    method("site", MustNotRead(), shape)
        before = runtime.artifact().artifact_id
        with self.assertRaises(ValueError):
            runtime.train_sequence_mse("site", [.2] * 18, OversizedTarget(), (2, 3), .1)
        self.assertEqual(runtime.artifact().artifact_id, before)
        pointwise = st.NativeGeometryRuntime(artifact("wave"))
        flat = pointwise.forward("site", [.2] * 18, 6)
        context = pointwise.forward_sequence("site", [.2] * 18, (2, 3))
        self.assertIsNone(flat.sequence_shape)
        self.assertEqual(context.sequence_shape, (2, 3, 3))
        self.assertEqual(flat.output, context.output)
        self.assertEqual(flat.vjp([1.] * 18), context.vjp([1.] * 18))

    def test_sequence_causality_and_cross_batch_isolation(self):
        values = [(i * 7 % 19) * .05 - .3 for i in range(18)]
        for kind in ("causal", "gated"):
            runtime = st.NativeGeometryRuntime(artifact(kind))
            batch = runtime.forward_sequence("site", values, (2, 3))
            changed = values[:6] + [v + 3. for v in values[6:]]
            self.assertEqual(batch.output[:6], runtime.forward_sequence("site", changed, (2, 3)).output[:6])
            for index in range(2):
                separate = runtime.forward_sequence("site", values[9 * index:9 * (index + 1)], (1, 3))
                self.assertEqual(separate.output, batch.output[9 * index:9 * (index + 1)])
            seed = [.2, -.4, .1] + [0.] * 15
            self.assertEqual(batch.vjp(seed)[0][3:], [0.] * 15)
            buffers = batch.vjp_buffer(array("f", seed))
            self.assertEqual(list(memoryview(buffers[0]).cast("f")), batch.vjp(seed)[0])

    def test_invalid_updates_leave_artifact_unchanged(self):
        runtime = st.NativeGeometryRuntime(artifact())
        batch = runtime.forward("site", [.2] * 6, 2)
        before = runtime.artifact().artifact_id
        for rate in (True, 0., -1., float("nan"), float("inf")):
            with self.subTest(rate=rate), self.assertRaises((TypeError, ValueError)):
                runtime.step_sgd(batch, [1.] * 6, rate)
            self.assertEqual(runtime.artifact().artifact_id, before)
        for direction in ([1.], [float("nan")] * 6, [float("inf")] * 6, [True] * 6):
            with self.assertRaises((ValueError, TypeError)):
                runtime.step_sgd(batch, direction, .1)
            self.assertEqual(runtime.artifact().artifact_id, before)
        with self.assertRaises(ValueError):
            runtime.train_mse("site", [.2] * 6, [float("nan")] * 6, 2, .1)
        self.assertEqual(runtime.artifact().artifact_id, before)
        with self.assertRaises(ValueError):
            runtime.step_sgd(batch, [1e30] * 6, 1e30)
        self.assertEqual(runtime.artifact().artifact_id, before)

    def test_all_frozen_and_unsupported_recipes_fail_without_fallback(self):
        state = json.loads(artifact().state_json())
        for tensor in state["placements"][0]["tensors"]:
            tensor["requires_grad"] = False
        runtime = st.NativeGeometryRuntime(st.GeometryAdapterArtifact.create_json(json.dumps(state)))
        before = runtime.artifact().artifact_id
        batch = runtime.forward("site", [.2] * 6, 2)
        with self.assertRaisesRegex(ValueError, "no trainable"):
            runtime.step_sgd(batch, [1.] * 6, .1)
        self.assertEqual(runtime.artifact().artifact_id, before)
        p = state["placements"][0]
        p["recipe"] = {"schema": "spiraltorch.topos_resonator_adapter.v1", "features": 3, "strength": .1,
                       "kernel": {"coupling": .1, "iterations": 1, "saturation": 1., "porosity": .05, "max_values": 64}}
        p["tensors"] = [t for t in p["tensors"] if t["name"] == "gate"]
        runtime = st.NativeGeometryRuntime(st.GeometryAdapterArtifact.create_json(json.dumps(state)))
        with self.assertRaisesRegex(ValueError, "currently supports"):
            runtime.forward("site", [.2] * 6, 2)

    def test_shape_and_value_budgets_precede_foreign_sequence_reads(self):
        class Oversized:
            def __len__(self): return 10**10
            def __getitem__(self, _): raise AssertionError("must reject without extraction")
        runtime = st.NativeGeometryRuntime(artifact())
        with self.assertRaises(ValueError):
            runtime.forward("site", Oversized(), 2)
        for rows in (True, -1, .5, 2**32):
            with self.assertRaises((TypeError, ValueError, OverflowError)):
                runtime.forward("site", [1.] * 6, rows)
        for values in (array("d", [1.] * 6), array("B", [0] * 24), memoryview(array("f", [1.] * 12))[::2]):
            with self.assertRaises((TypeError, ValueError, BufferError)):
                runtime.forward_buffer("site", values, 2)
        with self.assertRaisesRegex(ValueError, "unknown"):
            runtime.forward("missing", [], 0)
        empty = runtime.forward("site", [], 0)
        self.assertEqual(empty.output, [])
        self.assertEqual(empty.vjp([])[0], [])


if __name__ == "__main__":
    unittest.main()
