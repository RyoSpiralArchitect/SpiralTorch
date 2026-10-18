"""Public byte LM bindings: portable construction plus opt-in real GPU learning."""
import ast
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import unittest

import spiraltorch as st

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("byte_example", ROOT / "bindings/st-py/examples/resident_byte_geometry_lm.py")
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)
NAMES = (
    "ByteLmBatch", "ByteDecoderGeometryPlan", "ByteDecoderPlan", "ByteDecoderCheckpoint",
    "ByteDecoderCheckpointSnapshot", "ResidentByteBatch", "ResidentByteDecoder",
    "ByteDecoderForward", "ByteDecoderGradients",
)
MODES = ("none", "headwise", "mixture", "query_mixture") + example.METRICS


def batch():
    return st.nn.ByteLmBatch.from_windows([b"abcde", b"xyzxy"])


def values(tensor):
    return tensor.snapshot().read_values()


def bits(values):
    return struct.pack(f"<{len(values)}f", *values)


def gradients(model):
    forward = model.forward(model.prepare_batch(batch()))
    loss = forward.next_byte_loss()
    return loss, model.backward(forward, loss.prediction_gradient_tensor())


def step(model, data, rate=0.125):
    forward = model.forward(data)
    loss = forward.next_byte_loss()
    gradients = model.backward(forward, loss.prediction_gradient_tensor())
    model.sgd(gradients, rate).read()
    return values(loss.loss_tensor())[0]


class BytePortable(unittest.TestCase):
    def test_exports_and_stubs(self):
        namespace = {}
        exec("from spiraltorch.nn import *", namespace)
        classes = {n.name: n for n in ast.parse((ROOT / "bindings/st-py/spiraltorch/__init__.pyi").read_text()).body
                   if isinstance(n, ast.ClassDef)}
        exposed = {n.target.id for n in classes["_NnModule"].body if isinstance(n, ast.AnnAssign)}
        for name in NAMES:
            self.assertIn(name, st.nn.__all__)
            self.assertIs(namespace[name], getattr(st._rs.nn, name))
            self.assertIn("_Nn" + name, classes)
            self.assertIn(name, exposed)
            with self.assertRaises(TypeError):
                getattr(st.nn, name)()
        methods = {n.name for n in classes["_NnResidentByteDecoder"].body if isinstance(n, ast.FunctionDef)}
        for name in ("sgd_with_rates_clipped", "adamw_with_rates_clipped"):
            self.assertIn(name, methods)
            if st.wgpu_kernel_reports_available():
                self.assertTrue(callable(getattr(st.nn.ResidentByteDecoder, name)))

    def test_bytes_and_document_boundaries(self):
        data = st.nn.ByteLmBatch.from_documents([b"012345", bytes([0, 255, 128, 64])], [(0, 2), (1, 0)], 3)
        self.assertEqual(data.shape, (2, 3))
        self.assertEqual(data.input_bytes, b"234" + bytes([0, 255, 128]))
        self.assertEqual(data.target_bytes, b"345" + bytes([255, 128, 64]))
        for windows in ([], [b"a"], [b"ab", b"xyz"]):
            with self.assertRaises(ValueError):
                st.nn.ByteLmBatch.from_windows(windows)
        for windows in (["abc"], [[1, 2, 3]], [bytearray(b"abc")]):
            with self.assertRaises(TypeError):
                st.nn.ByteLmBatch.from_windows(windows)
        for selection, steps in (([(0, 4)], 2), ([(1, 0)], 1), ([(0, 0)], 0)):
            with self.assertRaises(ValueError):
                st.nn.ByteLmBatch.from_documents([b"012345"], selection, steps)
        for selection, steps in (([(False, 0)], 1), ([(0, True)], 1), ([(0, 0)], True)):
            with self.assertRaises(TypeError):
                st.nn.ByteLmBatch.from_documents([b"abc"], selection, steps)

    def test_all_geometry_modes_share_checkpoint_protocol(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                plan = example.make_plan(mode)
                self.assertEqual(plan.input_shape, (2, 4, 4))
                self.assertEqual(plan.output_shape, (2, 4, 256))
                self.assertEqual((plan.block_count, plan.position_capacity, plan.embedding_width), (1, 4, 4))
                checkpoint = plan.initial_checkpoint()
                restored = st.nn.ByteDecoderCheckpoint.from_json(checkpoint.to_json())
                self.assertEqual(restored.attempted_revision, 0)
                self.assertIsNone(restored.adamw_config)
                self.assertEqual(restored.to_json(), checkpoint.to_json())
                self.assertEqual(restored.plan().parameter_layout(), plan.parameter_layout())
                layout = plan.parameter_layout()
                self.assertEqual(layout["head"][1], layout["count"])
                if mode == "none":
                    self.assertIsNone(plan.causal_geometry)
                    self.assertIsNone(layout["geometry"])
                else:
                    self.assertEqual(plan.causal_geometry.contextual, mode == "query_mixture")
                    if mode in example.METRICS:
                        self.assertEqual(plan.causal_geometry.pair_metric, mode)
                    self.assertEqual(layout["geometry"]["all"][0], 2)

    def test_geometry_validation_and_immutable_builders(self):
        plan = example.make_plan("query_mixture")
        before = plan.initial_checkpoint().to_json()
        g = plan.causal_geometry
        with self.assertRaises(ValueError):
            g.with_pair_metric("fisher")
        with self.assertRaises(ValueError):
            g.with_mixture([example.METRICS[0]] * 2, [[0.] * 4], [[1.] * 4])
        with self.assertRaises(ValueError):
            g.with_mixture(list(example.METRICS), [[0.] * 8], [[0.] * 8])
        with self.assertRaises(ValueError):
            g.with_head_pair_metrics([[example.METRICS[0]]])
        with self.assertRaises(ValueError):
            g.with_query_projections([])
        with self.assertRaises(ValueError):
            plan.with_causal_geometry(g)
        with self.assertRaises(ValueError):
            plan.with_initial_geometry_raw_gains([[math.nan, 0.]])
        self.assertFalse(g.with_pair_metric(example.METRICS[0]).contextual)
        self.assertTrue(g.contextual)
        self.assertNotEqual(plan.with_initial_geometry_raw_gains([[0., 0.]]).initial_checkpoint().to_json(), before)
        self.assertEqual(plan.initial_checkpoint().to_json(), before)

    def test_checkpoint_bounds_and_rejections(self):
        original = example.make_plan().initial_checkpoint().to_json()
        with self.assertRaises(ValueError):
            st.nn.ByteDecoderCheckpoint.from_json(original, max_bytes=len(original.encode()) - 1)
        record = json.loads(original)
        for key, value in (("schema", "unknown"), ("update_rule", "adam"), ("extra", 1)):
            changed = dict(record, **{key: value})
            with self.assertRaises(ValueError):
                st.nn.ByteDecoderCheckpoint.from_json(json.dumps(changed))
        record["attempted_revision"] = "18446744073709551615"
        self.assertEqual(st.nn.ByteDecoderCheckpoint.from_json(json.dumps(record)).attempted_revision, 2**64 - 1)

    def test_cpu_only_execution_rejects_explicitly(self):
        if st.wgpu_kernel_reports_available():
            self.skipTest("requires CPU-only artifact")
        plan = example.make_plan()
        for call in (plan.compile_training_wgpu, plan.initial_checkpoint().restore_wgpu):
            with self.assertRaisesRegex(NotImplementedError, "wgpu"):
                call()


@unittest.skipUnless(os.environ.get("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS") == "1", "real WGPU opt-in")
class ByteGpu(unittest.TestCase):
    def test_clipped_sgd_uses_only_active_norm_and_retains_raw_gradients(self):
        model = example.make_plan().compile_training_wgpu()
        _, derivative = gradients(model)
        raw = [values(t) for t in derivative.parameter_gradient_tensors()]
        before = [values(t) for t in model.parameter_tensors()]
        active = model.parameter_layout()["head"][1] - 1
        rates = [0.0] * len(before)
        rates[active] = 0.125
        norm = math.sqrt(math.fsum(x * x for x in raw[active]))
        whole = math.sqrt(math.fsum(x * x for tensor in raw for x in tensor))
        limit = struct.unpack("<f", struct.pack("<f", 0.01))[0]
        self.assertGreater(norm, limit)
        self.assertGreater(whole, norm)
        wrong_norm_effect = max(abs(rates[active] * x * limit * (1 / norm - 1 / whole))
                                for x in raw[active])
        self.assertGreater(wrong_norm_effect, 3e-6)
        self.assertEqual(model.sgd_with_rates_clipped(derivative, rates, max_norm=limit).read(), 1)
        after = [values(t) for t in model.parameter_tensors()]
        for index, (old, new) in enumerate(zip(before, after)):
            if index != active:
                self.assertEqual(bits(old), bits(new))
            else:
                self.assertNotEqual(bits(old), bits(new))
                for original, gradient, actual in zip(old, raw[index], new):
                    expected = original - rates[index] * gradient * limit / norm
                    self.assertAlmostEqual(actual, expected, delta=3e-7 * max(1.0, abs(expected)))
        for original, retained in zip(raw, derivative.parameter_gradient_tensors()):
            self.assertEqual(bits(original), bits(values(retained)))

    def test_clipped_query_learning_resumes_from_one_identical_boundary(self):
        for adamw in (False, True):
            with self.subTest(adamw=adamw):
                model = example.make_plan().compile_training_wgpu()
                if adamw:
                    model.enable_adamw()
                layout = model.parameter_layout()
                frozen = {i for start, end in layout["geometry"]["query_projections"] for i in range(start, end)}
                initial = [values(t) for t in model.parameter_tensors()]

                def advance(owner, freeze=True):
                    loss, derivative = gradients(owner)
                    rate = 0.005 if adamw else 0.125
                    rates = [0.0 if freeze and i in frozen else rate for i in range(layout["count"])]
                    if adamw:
                        decays = [0.01] * len(rates)
                        for i in range(*layout["geometry"]["all"]):
                            decays[i] = 0.0
                        update = owner.adamw_with_rates_clipped(derivative, rates, max_norm=0.01,
                                                               weight_decays=decays)
                    else:
                        update = owner.sgd_with_rates_clipped(derivative, rates, max_norm=0.01)
                    return update, loss.loss_tensor()

                first_update, first_loss = advance(model)
                capture = model.checkpoint_snapshot()
                second_update, second_loss = advance(model)
                saved = capture.read().to_json()
                restored = st.nn.ByteDecoderCheckpoint.from_json(saved).restore_wgpu()
                self.assertEqual(restored.checkpoint_snapshot().read().to_json(), saved)
                resumed_update, resumed_loss = advance(restored)
                self.assertEqual(first_update.read(), 1)
                self.assertEqual(second_update.read(), 2)
                self.assertEqual(resumed_update.read(), 2)
                self.assertEqual(bits(values(second_loss)), bits(values(resumed_loss)))
                expected = model.checkpoint_snapshot().read().to_json()
                self.assertEqual(expected, restored.checkpoint_snapshot().read().to_json())
                final = [values(t) for t in model.parameter_tensors()]
                for i in frozen:
                    self.assertEqual(bits(initial[i]), bits(final[i]))
                geometry = range(*layout["geometry"]["all"])
                self.assertTrue(any(bits(initial[i]) != bits(final[i]) for i in geometry if i not in frozen))
                if adamw:
                    for i, slot in enumerate(json.loads(expected)["optimizer"]["slots"]):
                        self.assertEqual(slot["steps"], 0 if i in frozen else 2)
                self.assertTrue(math.isfinite(values(first_loss)[0]))
                for freeze in (False, True):
                    a, left = advance(model, freeze)
                    b, right = advance(restored, freeze)
                    self.assertEqual(a.read(), b.read())
                    self.assertEqual(bits(values(left)), bits(values(right)))
                    self.assertEqual(model.checkpoint_snapshot().read().to_json(),
                                     restored.checkpoint_snapshot().read().to_json())

    def test_clipped_adamw_moments_use_one_norm_across_active_tensors(self):
        model = example.make_plan().compile_training_wgpu()
        model.enable_adamw()
        beta1, beta2, _ = model.adamw_config
        _, derivative = gradients(model)
        raw = [values(t) for t in derivative.parameter_gradient_tensors()]
        before = [bits(values(t)) for t in model.parameter_tensors()]
        layout = model.parameter_layout()
        active = (layout["token_embedding"], layout["head"][1] - 1)
        rates = [0.0] * len(before)
        rates[active[0]], rates[active[1]] = 0.005, 0.01
        norm = math.sqrt(math.fsum(x * x for i in active for x in raw[i]))
        limit = struct.unpack("<f", struct.pack("<f", 0.01))[0]
        self.assertGreater(norm, limit)
        model.adamw_with_rates_clipped(derivative, rates, max_norm=limit,
                                      weight_decays=[0.02] * len(rates)).read()
        state = json.loads(model.checkpoint_snapshot().read().to_json())["optimizer"]
        for i, (slot, tensor) in enumerate(zip(state["slots"], model.parameter_tensors())):
            if i not in active:
                self.assertEqual(bits(values(tensor)), before[i])
                self.assertEqual(slot["steps"], 0)
                self.assertEqual(bits(slot["first"]), bytes(len(raw[i]) * 4))
                self.assertEqual(bits(slot["second"]), bytes(len(raw[i]) * 4))
                continue
            self.assertEqual(slot["steps"], 1)
            self.assertNotEqual(bits(values(tensor)), before[i])
            for gradient, first, second in zip(raw[i], slot["first"], slot["second"]):
                clipped = gradient * limit / norm
                expected_first = (1 - beta1) * clipped
                expected_second = (1 - beta2) * clipped * clipped
                self.assertAlmostEqual(first, expected_first, delta=3e-6 * max(abs(expected_first), 1e-30))
                self.assertAlmostEqual(second, expected_second, delta=3e-6 * max(abs(expected_second), 1e-30))

    def test_clipped_controls_and_ownership_reject_before_model_mutation(self):
        for adamw in (False, True):
            with self.subTest(adamw=adamw):
                model = example.make_plan().compile_training_wgpu()
                foreign = example.make_plan().compile_training_wgpu()
                if adamw:
                    model.enable_adamw()
                    foreign.enable_adamw()
                _, derivative = gradients(model)
                name = "adamw_with_rates_clipped" if adamw else "sgd_with_rates_clipped"
                update = getattr(model, name)
                rates = [0.005] * model.parameter_layout()["count"]
                original = model.checkpoint_snapshot().read().to_json()
                for limit in (0.0, -0.0, -1.0, math.nan, math.inf, 1e-100, 1e100):
                    with self.assertRaises(ValueError):
                        update(derivative, rates, max_norm=limit)
                for invalid in ([], [-0.1] * len(rates), [math.nan] * len(rates)):
                    with self.assertRaises(ValueError):
                        update(derivative, invalid, max_norm=0.01)
                if adamw:
                    for decays in ([], [-0.1] * len(rates), [math.inf] * len(rates)):
                        with self.assertRaises(ValueError):
                            update(derivative, rates, max_norm=0.01, weight_decays=decays)
                with self.assertRaises(TypeError):
                    update(derivative, rates)
                with self.assertRaises(ValueError):
                    getattr(foreign, name)(derivative, rates, max_norm=0.01)
                self.assertEqual(model.checkpoint_snapshot().read().to_json(), original)
                self.assertEqual(foreign.attempted_updates, 0)
                before = [bits(values(t)) for t in model.parameter_tensors()]
                self.assertEqual(update(derivative, [0.0] * len(rates), max_norm=2**-149).read(), 1)
                self.assertEqual([bits(values(t)) for t in model.parameter_tensors()], before)
                with self.assertRaises(ValueError):
                    update(derivative, rates, max_norm=0.01)
                self.assertEqual(model.attempted_updates, 1)

    def test_clipped_frozen_nonfinite_gradients_still_reject_atomically(self):
        for adamw in (False, True):
            with self.subTest(adamw=adamw):
                model = example.make_plan().compile_training_wgpu()
                if adamw:
                    model.enable_adamw()
                before = [bits(values(t)) for t in model.parameter_tensors()]
                old_optimizer = json.loads(model.checkpoint_snapshot().read().to_json()).get("optimizer")
                forward = model.forward(model.prepare_batch(batch()))
                huge = model.tensor_device().upload([2, 4, 256], [3e38] * 2048)
                bad = model.backward(forward, huge.mul(huge))
                update = model.adamw_with_rates_clipped if adamw else model.sgd_with_rates_clipped
                rejected = update(bad, [0.0] * len(before), max_norm=0.01)
                with self.assertRaises(ValueError):
                    rejected.read()
                self.assertEqual(model.attempted_updates, 1)
                self.assertEqual([bits(values(t)) for t in model.parameter_tensors()], before)
                self.assertEqual(json.loads(model.checkpoint_snapshot().read().to_json()).get("optimizer"), old_optimizer)
                _, good = gradients(model)
                self.assertEqual(update(good, [0.005] * len(before), max_norm=0.01).read(), 2)

    def test_adamw_geometry_history_and_exact_resume(self):
        for mode in ("none", "query_mixture"):
            with self.subTest(mode=mode):
                model = example.make_plan(mode).compile_training_wgpu()
                self.assertIsNone(model.adamw_config)
                for config in ({"beta1": 1.0}, {"epsilon": 1e-50}, {"beta2": math.nan}):
                    with self.assertRaises(ValueError):
                        model.enable_adamw(**config)
                model.enable_adamw()
                with self.assertRaises(ValueError):
                    model.enable_adamw()

                def update(owner, frozen=False):
                    data = owner.prepare_batch(batch())
                    forward = owner.forward(data)
                    loss = forward.next_byte_loss()
                    gradients = owner.backward(forward, loss.prediction_gradient_tensor())
                    layout = owner.parameter_layout()
                    rates = [0.005] * layout["count"]
                    decays = [0.01] * layout["count"]
                    if layout["geometry"]:
                        for i in range(*layout["geometry"]["all"]):
                            rates[i] = 0.0 if frozen else 0.005
                            decays[i] = 0.0
                    owner.adamw_with_rates(gradients, rates, weight_decays=decays).read()
                    return values(loss.loss_tensor())[0]

                first = update(model)
                capture = model.checkpoint_snapshot()
                second = update(model, True)
                saved = capture.read()
                self.assertEqual(saved.attempted_revision, 1)
                self.assertEqual(saved.adamw_config, model.adamw_config)
                record = json.loads(saved.to_json())
                self.assertEqual(record["schema"], "spiraltorch.nn.byte_decoder_checkpoint.v6")
                self.assertEqual(record["update_rule"], "adamw_zero_rate_freeze.v1")
                resumed = st.nn.ByteDecoderCheckpoint.from_json(saved.to_json()).restore_wgpu()
                self.assertEqual(update(resumed, True), second)
                for freeze in (False, True):
                    self.assertEqual(update(model, freeze), update(resumed, freeze))
                    self.assertEqual(model.checkpoint_snapshot().read().to_json(), resumed.checkpoint_snapshot().read().to_json())
                self.assertLess(second, first)
                data = model.prepare_batch(batch())
                forward = model.forward(data)
                gradient = model.backward(forward, forward.next_byte_loss().prediction_gradient_tensor())
                before = model.attempted_updates
                for bad in (1e-50, -0.1, math.nan, math.inf):
                    with self.assertRaises(ValueError):
                        model.adamw(gradient, bad)
                with self.assertRaises(ValueError):
                    model.adamw_with_rates(gradient, [])
                with self.assertRaises(ValueError):
                    model.sgd(gradient, 0.1)
                self.assertEqual(model.attempted_updates, before)
                self.assertEqual(model.adamw(gradient, 0.001).read(), before + 1)

    def test_all_modes_train_resume_and_retain(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                plan = example.make_plan(mode)
                model = plan.compile_training_wgpu()
                self.assertNotEqual(model.tensor_device().adapter_info()["device_type"], "Cpu")
                data = model.prepare_batch(batch())
                self.assertEqual(data.shape, (2, 4))
                self.assertEqual(values(data.targets_tensor()), [98., 99., 100., 101., 121., 122., 120., 121.])
                first = step(model, data)
                capture = model.checkpoint_snapshot()
                second = step(model, data)
                saved = capture.read()
                self.assertEqual(saved.attempted_revision, 1)
                with self.assertRaisesRegex(RuntimeError, "consumed"):
                    capture.read()
                restored = st.nn.ByteDecoderCheckpoint.from_json(saved.to_json()).restore_wgpu()
                other = restored.prepare_batch(batch())
                self.assertEqual(step(restored, other), second)
                third, third_restored = step(model, data), step(restored, other)
                self.assertEqual(third, third_restored)
                self.assertLess(third, first)
                self.assertEqual([values(t) for t in model.parameter_tensors()],
                                 [values(t) for t in restored.parameter_tensors()])
                forward = restored.forward(other)
                retained = forward.prediction_tensor()
                self.assertEqual(forward.parameter_revision, 3)
                if mode == "none":
                    self.assertIsNone(forward.geometry_pair_bias_tensor(0))
                else:
                    self.assertEqual(forward.geometry_pair_bias_tensor(0).shape, (2, 2, 4, 4))
                mixture = forward.geometry_mixture_weights_tensor(0)
                if mode in ("mixture", "query_mixture"):
                    self.assertEqual(mixture.shape, (2, 4) if mode == "mixture" else (2, 2, 4, 4))
                else:
                    self.assertIsNone(mixture)
                expected = values(retained)
                del model, restored, data, other, forward
                gc.collect()
                self.assertEqual(values(retained), expected)

    def test_query_controls_rates_external_biases_and_ownership(self):
        model = example.make_plan().compile_training_wgpu()
        device = model.tensor_device()
        data = model.prepare_batch(batch())
        z = device.upload([2, 2, 4], [0.03] * 16)
        pair = device.upload([2, 2, 4, 4], [0.02] * 64)
        forward = model.forward(data, z_biases=[z], pair_biases=[pair])
        loss = forward.next_byte_loss()
        seed = loss.prediction_gradient_tensor()
        full = model.backward(forward, seed)
        stopped = model.backward(forward, seed, query_pullback="stop_coordinates")
        self.assertEqual((full.query_pullback, stopped.query_pullback), ("full", "stop_coordinates"))
        self.assertNotEqual(values(full.embedding_output_gradient_tensor()), values(stopped.embedding_output_gradient_tensor()))
        self.assertEqual(full.z_bias_gradient_tensor(0).shape, z.shape)
        self.assertEqual(full.pair_bias_gradient_tensor(0).shape, pair.shape)
        with self.assertRaises(IndexError):
            full.z_bias_gradient_tensor(1)
        with self.assertRaises(ValueError):
            model.backward(forward, seed, query_pullback="typo")
        with self.assertRaises(ValueError):
            model.forward(data, pair_biases=[])
        good = model.backward(forward, seed)
        foreign = example.make_plan().compile_training_wgpu()
        with self.assertRaises(ValueError):
            foreign.sgd(good, 0.1)
        for rates in ([], [-0.1] * model.parameter_layout()["count"], [math.nan] * model.parameter_layout()["count"]):
            with self.assertRaises(ValueError):
                model.sgd_with_rates(good, rates)
        self.assertEqual(model.attempted_updates, 0)
        before = [values(t) for t in model.parameter_tensors()]
        self.assertEqual(model.sgd_with_rates(good, [0.] * len(before)).read(), 1)
        self.assertEqual([values(t) for t in model.parameter_tensors()], before)
        with self.assertRaises(ValueError):
            model.sgd(good, 0.1)
        with self.assertRaises(ValueError):
            model.backward(forward, seed)
        current = model.forward(data)
        self.assertIsNone(model.backward(current, current.next_byte_loss().prediction_gradient_tensor()).pair_bias_gradient_tensor(0))

    def test_nonfinite_seed_rejects_atomic_update_and_recovers(self):
        model = example.make_plan().compile_training_wgpu()
        data = model.prepare_batch(batch())
        before = [values(t) for t in model.parameter_tensors()]
        forward = model.forward(data)
        huge = model.tensor_device().upload([2, 4, 256], [3e38] * 2048)
        bad = model.backward(forward, huge.mul(huge))
        update = model.sgd(bad, 0.125)
        with self.assertRaises(ValueError):
            update.read()
        self.assertEqual(model.attempted_updates, 1)
        self.assertEqual([values(t) for t in model.parameter_tensors()], before)
        self.assertTrue(math.isfinite(step(model, data)))


if __name__ == "__main__":
    unittest.main()
