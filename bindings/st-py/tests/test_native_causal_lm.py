"""Installed whole-decoder execution with imports of foreign ML packages forbidden."""
import importlib.abc
import json
import math
import struct
import sys
import unittest


class NoForeignML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "transformers", "numpy", "accelerate", "safetensors"}:
            raise AssertionError("native causal LM must not import " + fullname)


if __name__ == "__main__":
    sys.meta_path.insert(0, NoForeignML())

import spiraltorch as st  # noqa: E402


def fixture(*, width=4, heads=2, positions=8, layers=2, vocabulary=11, raw_weights=False):
    config = {"model_type": "gpt2", "vocab_size": vocabulary, "n_positions": positions,
              "n_embd": width, "n_head": heads, "n_layer": layers}
    header, payload = {}, bytearray()

    def add(name, shape):
        count = math.prod(shape)
        values = [1. if "ln_" in name and name.endswith("weight")
                  else math.sin((i * 7 + len(header) * 3) * .23) * .15 for i in range(count)]
        start = len(payload)
        payload.extend(struct.pack("<" + "f" * count, *values))
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [start, len(payload)]}

    add("wte.weight", [vocabulary, width])
    add("wpe.weight", [positions, width])
    for layer in range(layers):
        for norm in ("ln_1", "ln_2"):
            for suffix in ("weight", "bias"):
                add(f"h.{layer}.{norm}.{suffix}", [width])
        for name, input_width, output_width in (("attn.c_attn", width, 3*width), ("attn.c_proj", width, width),
                                                ("mlp.c_fc", width, 4*width), ("mlp.c_proj", 4*width, width)):
            add(f"h.{layer}.{name}.weight", [input_width, output_width])
            add(f"h.{layer}.{name}.bias", [output_width])
    add("ln_f.weight", [width])
    add("ln_f.bias", [width])
    raw = json.dumps(header).encode()
    weights = struct.pack("<Q", len(raw)) + raw + payload
    return json.dumps(config), bytes(weights) if raw_weights else st.SafeTensorArchive.from_bytes(weights)


def artifact(model, *, path="transformer.h.0.mlp", gate=.5, trainable=False):
    return st.GeometryAdapterArtifact.create_json(json.dumps({
        "schema": "spiraltorch.geometry_adapter_artifact.v1", "base_model_identity": model.base_model_identity,
        "policy": "full_prefix_f32_no_cache.v1", "training": trainable, "placements": [{
            "path": path, "training": trainable,
            "recipe": {"schema": "spiraltorch.channel_affine_adapter.v1", "features": 4, "strength": .5, "max_values": 1024},
            "tensors": [{"name": name, "shape": [4], "values": values, "requires_grad": trainable and name == "gate", "role": "parameter"}
                        for name, values in [("gate", [gate, 0., -gate, gate]), ("bias", [0.] * 4)]],
        }],
    }))


class NativeCausalLmTests(unittest.TestCase):
    def setUp(self):
        self.config, self.weights = fixture()
        self.model = st.NativeCausalLm.from_gpt2(self.config, self.weights)

    def test_scalar_output_real_step_and_bit_exact_restore(self):
        from hashlib import sha256
        m = self.model.with_geometry(artifact(self.model))
        left, right = struct.pack("<4d", 1, 0, 0, 0), struct.pack("<4d", 0, 1, 0, 0)
        s = m.scalar_output_adapter(0, left, right)
        self.assertIsInstance(s, st.NativeCausalLmScalarOutput)
        pos, neg = [[1, 2, 3]], [[3, 2, 1]]
        kw = dict(learning_rate=16., max_step=.08, max_abs=4.)
        n, info = s.step_pairs(pos, neg, 2, 7, **kw)
        self.assertEqual((s.steps, s.coefficient, n.steps), (0, 0., 1))
        self.assertNotEqual(n.coefficient, 0.)
        self.assertEqual(len(info["pairs"]), 1)
        self.assertAlmostEqual(info["after"], n.coefficient, places=14)
        raw = n.checkpoint_json()
        pin = sha256(raw.encode()).hexdigest()
        restored = m.restore_scalar_output_adapter(raw, pin)
        self.assertEqual(n.step_pairs(pos, neg, 2, 7, **kw)[0].checkpoint_json(),
                         restored.step_pairs(pos, neg, 2, 7, **kw)[0].checkpoint_json())
        self.assertEqual(s.last_logits_f32le(pos[0]), struct.pack("<11f", *m.forward(pos[0], [1, 3], last_only=True).values()))
        with self.assertRaises(ValueError):
            self.model.restore_scalar_output_adapter(raw, pin)
        with self.assertRaises(ValueError):
            m.scalar_output_adapter(0, bytes(32), right)
        with self.assertRaises(ValueError):
            n.step_pairs(pos, neg, 2, 2, **kw)

    def test_scalar_search_reuses_pairs_and_refuses_without_advancing(self):
        s = self.model.scalar_output_adapter(0, struct.pack("<4d",1,0,0,0), struct.pack("<4d",0,1,0,0))
        p, n = [[1,2,3], [4,3,1]], [[3,2,1], [1,3,4]]
        options = dict(learning_rate=16., max_step=.08, max_abs=4., max_trials=3, min_decrease=5.)
        old = s.checkpoint_json()
        same, refusal = s.step_pairs_backtracking(p,n,2,7,**options)
        self.assertEqual((same.checkpoint_json(), refusal["step"], refusal["status"]), (old,0,"trials_exhausted"))
        self.assertEqual(refusal["baseline"], s.pair_objective(p,n,2,7))
        self.assertEqual(len(refusal["trials"]),3)
        fixed, direct = s.step_pairs(p,n,2,7, learning_rate=16., max_step=.08, max_abs=4.)
        self.assertEqual(refusal["trials"][0]["objective"],fixed.pair_objective(p,n,2,7))
        self.assertEqual(refusal["derivatives"],direct["pairs"])
        zero, flat = s.step_pairs_backtracking(p,p,2,7,**options)
        self.assertEqual((zero.steps,flat["status"],flat["trials"]),(0,"zero_gradient",[]))
        class NoRead:
            def __len__(self): raise AssertionError("inputs read before validating search bounds")
        for bad in ({"max_trials":0},{"max_trials":17},{"min_decrease":float("nan")},{"max_abs":0.0}):
            with self.assertRaises(ValueError):
                s.step_pairs_backtracking(NoRead(),NoRead(),2,7,**(options|bad))

    def test_scalar_search_recovers_an_overshoot_without_committing_rejected_candidates(self):
        from hashlib import sha256

        state = self.model.scalar_output_adapter(0, struct.pack("<4d", 1, 0, 0, 0), struct.pack("<4d", 0, 1, 0, 0))
        pos, neg = [[1, 2, 3], [4, 3, 1]], [[3, 2, 1], [1, 3, 4]]
        options = dict(learning_rate=1e7, max_step=1., max_abs=1., max_trials=3, min_decrease=1e-9)
        accepted, report = state.step_pairs_backtracking(pos, neg, 2, 7, **options)
        before = state.pair_objective(pos, neg, 2, 7)
        self.assertEqual(report["baseline"], before)
        self.assertEqual([t["coefficient"] for t in report["trials"]], [-1., -.5, -.25])
        self.assertTrue(all(t["objective"]["loss"] > before["loss"] for t in report["trials"][:2]))
        self.assertEqual((report["status"], accepted.steps, state.steps), ("accepted", 1, 0))
        self.assertLess(accepted.pair_objective(pos, neg, 2, 7)["loss"], before["loss"])
        checkpoint = accepted.checkpoint_json()
        restored = self.model.restore_scalar_output_adapter(checkpoint, sha256(checkpoint.encode()).hexdigest())
        self.assertEqual(restored.pair_objective(pos, neg, 2, 7), report["trials"][-1]["objective"])

    def test_indexed_output_directions_fit_and_open_a_bound_scalar(self):
        m = self.model.with_geometry(artifact(self.model))
        rows = [[1, 2, 3], [3, 2, 1], [4, 3, 1], [1, 3, 4]]
        left = struct.pack("<4d", 1, 0, 0, 0)
        bank = m.output_contrast_bank(rows, 2, 7, layer=0, left_f64le=left)
        self.assertIsInstance(bank, st.NativeCausalLmOutputContrastBank)
        self.assertEqual(len(bank), len(rows))
        self.assertEqual(bank.contrasts(), [m.next_token_contrast(r, 2, 7) for r in rows])
        ref = m.projection_contrast_pullback(rows[0], 2, 7, layer=0, site="output", basis="dense")
        self.assertEqual(bank.gradient_f64le(0), ref["weight_f64le"][:32])
        direction = bank.direction([[0, 1], [2, 3]], [0, 1])
        self.assertIsInstance(direction, st.NativeCausalLmOutputMarginDirection)
        self.assertIsNone(direction.certificate()["refusal"])
        self.assertEqual(direction.left_f64le(), left)
        self.assertEqual(len(direction.coordinates_f64le()), 32)
        s = direction.scalar_adapter(m)
        pos, neg = [rows[0], rows[0], rows[2]], [rows[1], rows[3], rows[3]]
        _, r = s.step_pairs_backtracking(pos, neg, 2, 7, learning_rate=1., max_step=.02,
                                        max_abs=1., max_trials=2, min_decrease=2.)
        self.assertEqual(r["gradient_prompt_evaluations"], 4)
        self.assertTrue(all(z["prompt_evaluations"] == 4 for z in [r["baseline"], *(t["objective"] for t in r["trials"])]))
        with self.assertRaises(ValueError):
            direction.scalar_adapter(self.model)
        refusal = bank.direction([[0, 1], [1, 0]], [0, 1])
        self.assertEqual(refusal.certificate()["refusal"], "unresolved")
        self.assertIsNone(refusal.coordinates_f64le())

    def test_indexed_output_admission_checks_roster_before_copying_ids(self):
        class Unread:
            def __len__(self): return 99
            def __getitem__(self, key): raise AssertionError("read inadmissible input")
        with self.assertRaises(ValueError):
            self.model.output_contrast_bank(Unread(), 2, 7, layer=0)
        with self.assertRaises(ValueError):
            self.model.output_contrast_bank([Unread()], 2, 7, layer=0)
        with self.assertRaises(ValueError):
            self.model.output_contrast_bank([[1],[1]], 2, 7, layer=0)
        bank = self.model.output_contrast_bank([[1],[2]], 2, 7, layer=0)
        self.assertEqual(len(bank.gradient_f64le(1)), 128)
        with self.assertRaises(ValueError):
            bank.direction([[1, 2], [0, 1]], [0, 1])

    def test_full_model_shapes_causality_and_batch_independence(self):
        model = self.model
        full = model.forward([1, 2, 3, 4, 5, 6], [2, 3])
        self.assertEqual(full.shape, [2, 3, 11])
        values = full.values()
        for i, row in enumerate(([1, 2, 3], [4, 5, 6])):
            self.assertEqual(values[i * 33:(i + 1) * 33], model.forward(row, [1, 3]).values())
        changed = model.forward([1, 2, 9, 4, 5, 10], [2, 3]).values()
        self.assertEqual(values[:22], changed[:22])
        self.assertEqual(values[33:55], changed[33:55])
        self.assertEqual(model.forward([1, 2, 3, 4, 5, 6], [2, 3], last_only=True).values(),
                         values[22:33] + values[55:66])
        self.assertIn("NativeCausalLm", st.__all__)

    def test_projection_pullback_dense_bytes_and_independent_finite_slope(self):
        m=self.model.with_readout_accumulator("f32_wide_backward")
        ids=[4,8,2,1];kw=dict(layer=0,site="query",basis="dense")
        r=m.projection_contrast_pullback(ids,2,7,**kw)
        self.assertEqual(r["contrast"],m.next_token_contrast(ids,2,7))
        self.assertEqual((r["layer"],r["site"],r["basis"],r["width"],r["readout_accumulator"]),(0,"query","dense",4,"f32_wide_backward"))
        bias=struct.unpack("<4d",r["bias_f64le"]);w=struct.unpack("<16d",r["weight_f64le"])
        diag=m.projection_contrast_pullback(ids,2,7,layer=0,site="query")
        self.assertEqual(tuple(w[i*5] for i in range(4)),struct.unpack("<4d",diag["weight_f64le"]))
        length=math.hypot(*bias,*w);self.assertGreater(length,1e-5)
        def score(h):
            trial=m.projection_perturbed_contrast(ids,2,7,**kw,
                bias_f32le=struct.pack("<4f",*(h*x/length for x in bias)),weight_f32le=struct.pack("<16f",*(h*x/length for x in w)))
            return trial["target_minus_alternative"]
        h=.003
        self.assertAlmostEqual((score(h)-score(-h))/2,h*length,delta=8e-7+.015*h*length)
        self.assertEqual(r["contrast"],m.next_token_contrast(ids,2,7))

    def test_projection_rejects_invalid_intervention_before_copying_tokens(self):
        class Unread:
            def __len__(self):return 2
            def __getitem__(self,i):raise AssertionError("copied prompt before admission")
        for kw in (dict(layer=2,site="key"),dict(layer=0,site="not-a-site"),dict(layer=0,site="value",basis="all")):
            with self.assertRaises(ValueError):self.model.projection_contrast_pullback(Unread(),2,7,**kw)
        for b,w in ((bytes(15),bytes(16)),(bytes(16),bytes(64)),(struct.pack("<4f",0,0,float("nan"),0),bytes(16))):
            with self.assertRaises(ValueError):
                self.model.projection_perturbed_contrast(Unread(),2,7,layer=0,site="key",bias_f32le=b,weight_f32le=w)

    def test_loss_shift_greedy_and_model_identity(self):
        model = self.model
        logits = model.forward([1, 2, 4, 5], [2, 2]).values()
        loss = 0.
        for i, label in enumerate([2, 3, 5, 6]):
            row = logits[i * 11:(i + 1) * 11]
            maximum = max(row)
            loss += maximum - row[label] + math.log(sum(math.exp(x - maximum) for x in row))
        self.assertAlmostEqual(model.next_token_loss([1, 2, 3, 4, 5, 6], [2, 3]), loss / 4, places=6)
        last = model.forward([1, 2, 3], [1, 3], last_only=True).values()
        self.assertEqual(model.next_token_ids([1, 2, 3], [1, 3]), [max(range(11), key=last.__getitem__)])
        self.assertEqual(st.NativeCausalLm.from_gpt2(self.config, self.weights).base_model_identity, model.base_model_identity)
        self.assertTrue(self.weights.content_id.startswith("sha256:"))
        self.assertEqual(json.loads(model.config_json())["vocabulary"], 11)

    def test_next_token_score_and_saved_f32le_share_native_ranking(self):
        model = self.model.with_geometry(artifact(self.model))
        prompt = list(range(8))  # No spare position to append the target.
        values = model.forward(prompt, [1, 8], last_only=True).values()
        self.assertNotEqual(values, self.model.forward(prompt, [1, 8], last_only=True).values())
        wide = model.with_readout_accumulator("f64")
        wide_raw = struct.pack("<11f", *wide.forward(prompt, [1, 8], last_only=True).values())
        self.assertEqual(wide.next_token_score(prompt, 0), st.score_token_logits_f32le(wide_raw, 0))
        raw = struct.pack("<11f", *values)
        order = sorted(range(11), key=lambda i: (-values[i], i))
        self.assertIn("score_token_logits_f32le", st.__all__)
        for target in (order[0], order[1], order[-1]):
            s = model.next_token_score(prompt, target_id=target)
            self.assertEqual(s, st.score_token_logits_f32le(raw, target))
            other = next(i for i in order if i != target)
            self.assertEqual((s["target_id"], s["vocabulary"], s["target_rank"], s["strongest_rival_id"], s["greedy_id"]),
                             (target, 11, order.index(target) + 1, other, order[0]))
            self.assertEqual(s["target_minus_best_rival"], values[target] - values[other])
            peak = max(values)
            self.assertAlmostEqual(s["nll"], peak - values[target] + math.log(math.fsum(math.exp(v-peak) for v in values)), places=13)
        tied = st.score_token_logits_f32le(struct.pack("<3f", -0., 0., -1.), 1)
        self.assertEqual((tied["target_rank"], tied["strongest_rival_id"], tied["greedy_id"]), (2, 0, 0))
        tiny = st.score_token_logits_f32le(struct.pack("<2f", 0., -60.), 0)
        self.assertAlmostEqual(tiny["nll"] / math.log1p(math.exp(-60.)), 1., places=13)
        for data, target in [(raw[:-1], 0), (raw[:4], 0), (b"\0" * (4 * (2**20 + 1)), 0),
                             (raw, True), (raw, 11), (struct.pack("<2f", 0., float("nan")), 0), (list(raw), 0)]:
            with self.assertRaises((ValueError, TypeError)):
                st.score_token_logits_f32le(data, target)

        class Unread:
            def __len__(self):
                return 1
            def __getitem__(self, i):
                raise AssertionError("invalid target must be rejected before reading IDs")
        with self.assertRaises(ValueError):
            model.next_token_score(Unread(), 11)
        with self.assertRaises(TypeError):
            model.next_token_score([1], True)

    def test_next_token_contrast_is_prompt_only_and_not_a_forced_choice(self):
        prompt = list(range(8))
        full = self.model.forward(prompt, [1, 8]).values()[-11:]
        greedy = max(range(11), key=full.__getitem__)
        a, b = (greedy + 1) % 11, (greedy + 2) % 11
        c = self.model.next_token_contrast(prompt, target_id=a, alternative_id=b)
        self.assertEqual(c, dict(prompt_tokens=8, target_id=a, alternative_id=b, greedy_id=greedy,
                                target_logit=full[a], alternative_logit=full[b],
                                target_minus_alternative=full[a] - full[b]))
        for ids, a, b in [([], 1, 2), ([11], 1, 2), ([1], 1, 1), ([1], 0, 11), ([1], True, 2)]:
            with self.subTest(ids=ids, a=a, b=b), self.assertRaises((ValueError, TypeError)):
                self.model.next_token_contrast(ids, a, b)

        class TooLong:
            def __len__(self):
                return 9

            def __getitem__(self, index):
                raise AssertionError("read a token before validating the prompt budget")

        with self.assertRaises(ValueError):
            self.model.next_token_contrast(TooLong(), 1, 2)

        class Unread(TooLong):
            def __len__(self):
                return 1

        for a, b in [(2, 2), (11, 0), (1, 11)]:
            with self.assertRaisesRegex(ValueError, "distinct in-vocabulary"):
                self.model.next_token_contrast(Unread(), a, b)

    def test_geometry_changes_logits_not_the_base_model(self):
        model = self.model
        before = model.forward([1, 2, 3], [1, 3]).values()
        for path in ("transformer.h.0", "transformer.h.1.attn", "transformer.h.0.mlp"):
            identity = model.with_geometry(artifact(model, path=path, gate=0.))
            self.assertEqual(identity.forward([1, 2, 3], [1, 3]).values(), before)
            state = artifact(model, path=path)
            adapted = model.with_geometry(state)
            self.assertEqual(adapted.geometry_artifact_id, state.artifact_id)
            self.assertNotEqual(adapted.forward([1, 2, 3], [1, 3]).values(), before)
        self.assertIsNone(model.geometry_artifact_id)
        self.assertEqual(model.forward([1, 2, 3], [1, 3]).values(), before)
        with self.assertRaisesRegex(ValueError, "unsupported native geometry placement"):
            model.with_geometry(artifact(model, path="transformer.h.99.mlp"))

    def test_rival_training_is_opt_in_geometry_only_and_admits_before_copying(self):
        model = self.model.with_lora(0, 2, 3.).with_geometry(
            artifact(self.model, path="transformer.h.0.attn", trainable=True))
        ids, labels = [1, 2, 3, 4], [-100, -100, -100, 4]
        ce = model.adaptation_loss_gradients(ids, [1, 4], mode="geometry", labels=labels)
        zero = model.geometry_competition_gradients(ids, [1, 4], labels=labels, rivals=3, mix=0.)
        self.assertIsNone(ce.competition_report())
        self.assertEqual((zero.loss, zero.parameter_gradients()), (ce.loss, ce.parameter_gradients()))
        grads = model.geometry_competition_gradients(ids, [1, 4], labels=labels, rivals=3, mix=.5)
        r = grads.competition_report()
        self.assertEqual((r["target_tokens"], r["rivals"], r["mix"]), (1, 3, .5))
        self.assertAlmostEqual(grads.loss, .5 * (r["full_ce"] + r["rival_ce"]), places=6)
        self.assertEqual(set(grads.parameter_gradients()), {"transformer.h.0.attn"})
        updated = model.step_adaptation_sgd(grads, "geometry", .1)
        self.assertNotEqual(updated.geometry_artifact_id, model.geometry_artifact_id)
        self.assertEqual(updated.lora_to_json(), model.lora_to_json())
        with self.assertRaises(ValueError):
            model.step_adaptation_sgd(grads, "joint", .1)

        class Unread:
            def __len__(self):
                raise AssertionError("read rejected token input")
        for k, mix in [(0, .5), (11, .5), (True, .5), (2, True), (2, float("nan")), (2, 1.1)]:
            with self.subTest(k=k, mix=mix), self.assertRaises((TypeError, ValueError)):
                model.geometry_competition_gradients(Unread(), [1, 4], rivals=k, mix=mix)
        with self.assertRaises(ValueError):
            self.model.geometry_competition_gradients(Unread(), [1, 4], rivals=2, mix=.5)

    def test_declared_rivals_use_only_named_classes_and_geometry(self):
        model = self.model.with_lora(0, 2, 3.).with_geometry(
            artifact(self.model, path="transformer.h.0.attn", trainable=True))
        ids, labels = [1, 2, 3, 4, 5, 6], [-100, 2, -100, -100, -100, 6]
        def grad(contrasts, mix):
            return model.geometry_declared_rival_gradients(ids, [2, 3], contrasts=contrasts, mix=mix, labels=labels)
        ce = model.adaptation_loss_gradients(ids, [2, 3], mode="geometry", labels=labels)
        zero = grad([(5, 6, [0, 3])], 0.)
        self.assertEqual((zero.loss, zero.parameter_gradients()), (ce.loss, ce.parameter_gradients()))
        self.assertIsNone(ce.declared_rival_report())
        g = grad([(5, 6, [0, 3])], 1.)
        r = g.declared_rival_report()
        logits = model.forward([4, 5], [1, 2], last_only=True).values()
        restricted = math.log1p(sum(math.exp(logits[j] - logits[6]) for j in (0, 3)))
        self.assertAlmostEqual(r["declared_ce"], restricted, places=12)
        self.assertAlmostEqual(g.loss, (model.next_token_score([1], 2)["nll"] + restricted)/2, places=6)
        self.assertEqual((r["target_tokens"], r["contrasted_tokens"], r["rival_ids"]), (2, 1, 2))
        self.assertIsNone(g.competition_report())
        nxt = model.step_adaptation_sgd(g, "geometry", .1)
        self.assertNotEqual(nxt.geometry_artifact_id, model.geometry_artifact_id)
        self.assertEqual(nxt.lora_to_json(), model.lora_to_json())
        class Unread:
            def __len__(self): raise AssertionError("should not read windows on invalid policy")
        for bad, mix in [([], True), ([], float("nan")), ([(3, 4, [2])], .5), ([(5, 6, [True])], .5),
                         ([(5, 6, [6])], .5), ([(5, 6, [3, 1])], .5), ([(5, 6, [1])]*2, .5)]:
            with self.subTest(bad=bad, mix=mix), self.assertRaises((TypeError, ValueError)):
                model.geometry_declared_rival_gradients(Unread(), [2, 3], contrasts=bad, mix=mix)
        for bad in [[(5, 2, [1])], [(2, 3, [1])]]:
            with self.assertRaises(ValueError): grad(bad, .5)

    def test_mixture_probe_pins_forward_and_exports_native_boundary_bytes(self):
        site = "transformer.h.1.attn"
        m = self.model.with_lora(1, 2, 3.).with_geometry(artifact(self.model, path=site, trainable=True))
        prompt, target, rival = [3, 5, 2], 7, 4
        p = m.last_attention_mixture_probe(prompt, target, rival, mix=.5)
        self.assertEqual(p["logits"], bytes(m.forward(prompt, [1,3], last_only=True).values_buffer()))
        self.assertEqual((p["forward_readout"],p["encoding"],p["site"]), ("f32","little_endian_f32",site))
        self.assertEqual(p["geometry_artifact_id"], m.geometry_artifact_id)
        obj = p["objectives"]
        for arm in ("ce","declared","mixed"):
            ids, labels = prompt+[target], [-100]*3+[target]
            g = (m.adaptation_loss_gradients(ids, [1,4], labels=labels, mode="geometry") if arm=="ce" else
                 m.geometry_declared_rival_gradients(ids, [1,4], labels=labels, contrasts=[(3,target,[rival])], mix=1. if arm=="declared" else .5))
            self.assertEqual(g.loss, obj[arm]["loss"])
            self.assertEqual(len(obj[arm]["d_logits"]), 4*11)
            self.assertEqual(set(obj[arm]["paths"]), {"f32","f64"})
            factual = obj[arm]["paths"]["f32"]
            self.assertEqual(factual["parameters"], {k:struct.pack("<"+"f"*len(v),*v) for k,v in g.parameter_gradients()[site].items()})
            self.assertEqual(len(factual["boundaries"]["attention_output_upstream"]), 4*4)
        class Unread:
            def __len__(self): return 2
            def __getitem__(self, i): raise AssertionError("should not copy inadmissible IDs")
        for a,b,mix in [(2,2,.5),(True,4,.5),(0,11,.5),(0,1,True),(0,1,float("nan"))]:
            with self.assertRaises((TypeError,ValueError)):
                m.last_attention_mixture_probe(Unread(), a, b, mix=mix)
        with self.assertRaises(ValueError):
            self.model.last_attention_mixture_probe(Unread(), 0, 1, mix=.5)

    def test_gradient_alignment_requires_the_same_point_and_selected_parameters(self):
        m = self.model.with_lora(0, 2, 3.).with_geometry(
            artifact(self.model, path="transformer.h.0.attn", trainable=True))
        a = m.adaptation_loss_gradients([1, 2, 3], [1, 3], mode="geometry")
        b = m.adaptation_loss_gradients([4, 5], [1, 2], mode="geometry")
        r = a.alignment_with(b)
        self.assertEqual((r["left_target_tokens"], r["right_target_tokens"]), (2, 1))
        self.assertEqual(r["geometry_artifact_id"], m.geometry_artifact_id)
        x, y = a.parameter_gradients(), b.parameter_gradients()
        pairs = [(i, j) for path, group in x.items() for name, vals in group.items()
                 for i, j in zip(vals, y[path][name], strict=True)]
        self.assertEqual(r["total"]["elements"], len(pairs))
        self.assertAlmostEqual(r["total"]["dot"], math.fsum(i*j for i,j in pairs), places=12)
        self.assertAlmostEqual(r["total"]["right_loss_slope"], -math.fsum(j*(i+j)/2 for i,j in pairs), places=12)
        self.assertEqual(r["total"]["left_loss_slope"], b.alignment_with(a)["total"]["right_loss_slope"])
        with self.assertRaises(ValueError):
            a.alignment_with(m.adaptation_loss_gradients([4, 5], [1, 2], mode="lora"))
        updated = m.step_adaptation_sgd(a, "geometry", .1)
        with self.assertRaises(ValueError):
            a.alignment_with(updated.adaptation_loss_gradients([4, 5], [1, 2], mode="geometry"))

    def test_paired_geometry_training_matches_independent_steps_after_the_arms_diverge(self):
        state = json.loads(artifact(self.model, path="transformer.h.1.attn", trainable=True).state_json())
        site = state["placements"][0]
        site["recipe"] = dict(schema="spiraltorch.wave_stage_adapter.v1", features=4, strength=.25, max_values=1024,
                              stage=dict(kind="projection", curvature=-1., log_radius=0.))
        for t in site["tensors"]:
            t["requires_grad"] = True
        model = self.model.with_lora(1, 2, 3.)
        lora = json.loads(model.lora_to_json())
        lora["b"] = [.1, -.2, .15, .05, -.1, .2, .1, -.15]
        ce = mixed = model.with_lora_json(json.dumps(lora)).with_geometry(st.GeometryAdapterArtifact.create_json(json.dumps(state)))
        frozen = ce.lora_to_json()
        for i in range(4):
            ids = [1+i, 2, 4, 3, 8-i, 6, 5, 7]
            shape = [2, 4]
            labels = [-100, -100, -100, 3, -100, -100, 5, -100] if i % 2 == 0 else None
            cg = ce.adaptation_loss_gradients(ids, shape, labels=labels, mode="geometry")
            mg = mixed.geometry_competition_gradients(ids, shape, labels=labels, rivals=3, mix=.5)
            ref = ce.step_adaptation_sgd(cg, "geometry", .75)
            direction = mixed.step_adaptation_sgd(mg, "geometry", .75)
            dist = ce.geometry_artifact().trainable_delta_l2(ref.geometry_artifact())
            expected, receipt = mixed.geometry_artifact().match_trainable_delta(direction.geometry_artifact(), dist)
            one, single_receipt = mixed.step_geometry_sgd_at_distance(mg, learning_rate=.75, target_l2=dist)
            a, b, report = ce.geometry_competition_pair_step(mixed, ids, shape, labels=labels, learning_rate=.75, rivals=3, mix=.5)
            if i == 2:
                def restore(handle):
                    art = handle.geometry_artifact()
                    checked = st.GeometryAdapterArtifact.from_json(art.to_json(), art.artifact_id, handle.base_model_identity)
                    return self.model.with_lora_json(frozen).with_geometry(checked)
                ra, rb, rr = restore(ce).geometry_competition_pair_step(
                    restore(mixed), ids, shape, labels=labels, learning_rate=.75, rivals=3, mix=.5)
                self.assertEqual((ra.geometry_artifact_id, rb.geometry_artifact_id, rr),
                                 (a.geometry_artifact_id, b.geometry_artifact_id, report))
            self.assertEqual(a.geometry_artifact().to_json(), ref.geometry_artifact().to_json())
            self.assertEqual(b.geometry_artifact().to_json(), expected.to_json())
            self.assertEqual((one.geometry_artifact_id, single_receipt), (expected.artifact_id, receipt))
            self.assertEqual((report["mixed_step"], report["competition"]), (receipt, mg.competition_report()))
            self.assertEqual((report["target_tokens"], report["shared_prefix_layers"]), (cg.target_tokens, 1))
            self.assertEqual(report["reference_pre_geometry_id"], ce.geometry_artifact_id)
            self.assertEqual(report["reference_post_geometry_id"], a.geometry_artifact_id)
            self.assertEqual(report["pre_arm_separation_l2"] > 0., i > 0)
            self.assertEqual((a.lora_to_json(), b.lora_to_json()), (frozen, frozen))
            self.assertAlmostEqual(receipt["realized_l2"] / dist, 1., delta=1e-4)
            ce, mixed = a, b

    def test_complete_line_objective_uses_stop_and_independent_full_vocab_prefixes(self):
        model = self.model.with_lora(1, 2, 3.).with_geometry(
            artifact(self.model, path="transformer.h.1.attn", trainable=True)
        ).with_readout_accumulator("f32_wide_backward")
        prompt, menu, gold, stop = [1, 3], [[2], [2, 4], [2, 4, 7], [2, 5]], 1, 0
        costs = []
        for line in menu:
            ids, parts = prompt.copy(), []
            for target in line+[stop]:
                parts.append(model.next_token_score(ids, target)["nll"])
                ids.append(target)
            costs.append(math.fsum(parts))
        conditional = math.log(math.fsum(math.exp(costs[gold]-v) for v in costs))
        for mix in (0., .5, 1.):
            kw = dict(gold_index=gold, stop_id=stop, mix=mix)
            report = model.line_menu_loss(prompt, menu, **kw)
            grad = model.geometry_line_menu_gradients(prompt, menu, **kw)
            self.assertEqual(report, grad.line_menu_report())
            self.assertIsNone(grad.declared_rival_report())
            self.assertEqual((grad.target_tokens, grad.window_shape), (12, [4,6]))
            self.assertEqual((report["shared_content_tokens"], report["target_counts"]), (1, [2,3,4,3]))
            self.assertAlmostEqual(report["loss"], (1-mix)*costs[gold]/3+mix*conditional, delta=2e-6)
            self.assertAlmostEqual(math.fsum(report["menu_probabilities"]), 1., delta=1e-12)
            for x,y in zip(report["line_nlls"], costs):
                self.assertAlmostEqual(x,y,delta=2e-6)
        self.assertEqual(self.model.line_menu_loss(prompt, menu, **kw)["shared_content_tokens"], 1)
        nxt = model.step_adaptation_sgd(grad,"geometry",.25)
        self.assertEqual(nxt.lora_to_json(),model.lora_to_json())
        self.assertLess(nxt.line_menu_loss(prompt,menu,**kw)["menu_nll"],report["menu_nll"])
        with self.assertRaises(ValueError):
            nxt.step_adaptation_sgd(grad,"geometry",.25)

    def test_absolute_terminated_set_needs_no_gold_and_updates_only_geometry(self):
        model = self.model.with_lora(1, 2, 3.).with_geometry(
            artifact(self.model, path="transformer.h.1.attn", trainable=True)
        ).with_readout_accumulator("f32_wide_backward")
        prompt, lines, stop = [1, 3], [[2], [2, 4], [2, 4, 7], [2, 5]], 0
        costs = []
        for line in lines:
            prefix, terms = prompt.copy(), []
            for token in [*line, stop]:
                terms.append(model.next_token_score(prefix, token)["nll"])
                prefix.append(token)
            costs.append(math.fsum(terms))
        r = model.line_set_loss(prompt, lines, stop_id=stop)
        g = model.geometry_line_set_gradients(prompt, lines, stop_id=stop)
        self.assertEqual(g.line_set_report(), r)
        self.assertIsNone(g.line_menu_report())
        self.assertNotIn("gold_index", r)
        self.assertEqual((r["target_counts"], r["prefix_nodes"], r["decoder_lanes"], g.target_tokens, g.window_shape),
                         ([2,3,4,3], 5, 2, 12, [4,6]))
        self.assertAlmostEqual(r["set_nll"], -math.log(math.fsum(math.exp(-v) for v in costs)), delta=2e-6)
        for j, line in enumerate(lines):
            self.assertAlmostEqual(model.line_set_loss(prompt, [line], stop_id=stop)["set_nll"], costs[j], delta=2e-6)
            c = model.line_menu_loss(prompt, lines, stop_id=stop, gold_index=j, mix=1.)
            self.assertAlmostEqual(r["set_nll"] + c["menu_nll"], costs[j], delta=2e-6)
        nxt = model.step_adaptation_sgd(g, "geometry", .25)
        self.assertLess(nxt.line_set_loss(prompt, lines, stop_id=stop)["set_nll"], r["set_nll"])
        self.assertEqual((nxt.base_model_identity, nxt.lora_to_json()), (model.base_model_identity, model.lora_to_json()))
        with self.assertRaises(ValueError):
            nxt.step_adaptation_sgd(g, "geometry", .25)

    def test_subset_ranking_matches_summed_alias_probabilities_and_seals_geometry_only(self):
        m = self.model.with_lora(1,2,3.).with_geometry(artifact(self.model, path="transformer.h.1.attn", trainable=True))
        prompt, lines, accepted = [1,3], [[2],[2,4],[2,4,7],[2,5]], [0,2]
        opts = dict(accepted_indices=accepted, stop_id=0)
        r = m.line_subset_loss(prompt,lines,**opts)
        g = m.geometry_line_subset_gradients(prompt,lines,**opts)
        self.assertEqual(g.line_subset_report(),r)
        self.assertIsNone(g.line_menu_report())
        self.assertEqual((r["prefix_nodes"],r["decoder_lanes"],g.target_tokens,g.window_shape), (5,2,12,[4,6]))
        menu = m.line_menu_loss(prompt,lines,gold_index=0,stop_id=0,mix=1.)
        absolute = m.line_set_loss(prompt,[lines[i] for i in accepted],stop_id=0)
        self.assertAlmostEqual(r["accepted_set_nll"],absolute["set_nll"],delta=2e-6)
        self.assertAlmostEqual(r["conditional_nll"],-math.log(sum(menu["menu_probabilities"][i] for i in accepted)),delta=2e-6)
        ga=m.geometry_line_set_gradients(prompt,[lines[i] for i in accepted],stop_id=0).parameter_gradients()
        gn=m.geometry_line_set_gradients(prompt,[line for i,line in enumerate(lines) if i not in accepted],stop_id=0).parameter_gradients()
        for path,params in g.parameter_gradients().items():
            for name,vs in params.items():
                for v,a,n in zip(vs,ga[path][name],gn[path][name]):
                    self.assertAlmostEqual(v,r["rejected_probability_given_menu"]*(a-n),delta=4e-7)
        nxt = m.step_adaptation_sgd(g,"geometry",.25)
        self.assertLess(nxt.line_subset_loss(prompt,lines,**opts)["conditional_nll"],r["conditional_nll"])
        self.assertEqual((m.base_model_identity,m.lora_to_json()),(nxt.base_model_identity,nxt.lora_to_json()))
        with self.assertRaises(ValueError):
            nxt.step_adaptation_sgd(g,"geometry",.25)

    def test_geometry_cohort_one_vote_per_handle_and_realized_predictions(self):
        m=self.model.with_lora(1,2,3.).with_geometry(artifact(self.model,path="transformer.h.1.attn",trainable=True))
        a=m.geometry_line_subset_gradients([1,3],[[2],[2,4],[3,4]],accepted_indices=[0,2],stop_id=0)
        b=m.geometry_line_set_gradients([1],[[2]],stop_id=0)
        members=[a,b,b]
        next_model,r=m.step_geometry_cohort_sgd(members,.125,target_l2=.002)
        self.assertEqual((r["members"],r["total_target_tokens"],r["trainable_values"]),(3,a.target_tokens+2*b.target_tokens,4))
        self.assertAlmostEqual(r["mean_input_loss"],(a.loss+2*b.loss)/3,places=12)
        self.assertEqual(r["pre_geometry_id"],m.geometry_artifact_id)
        self.assertEqual(r["post_geometry_id"],next_model.geometry_artifact_id)
        self.assertEqual(next_model.lora_to_json(),m.lora_to_json())
        def coords(handle):
            ts=json.loads(handle.geometry_artifact().state_json())["placements"][0]["tensors"]
            return next(t["values"] for t in ts if t["name"]=="gate")
        delta=[y-x for x,y in zip(coords(m),coords(next_model))]
        self.assertAlmostEqual(math.sqrt(math.fsum(x*x for x in delta)),.002,delta=2e-7)
        vs=[g.parameter_gradients()["transformer.h.1.attn"]["gate"] for g in members]
        self.assertAlmostEqual(r["mean_gradient_l2"],math.sqrt(math.fsum((math.fsum(row)/3)**2 for row in zip(*vs))),delta=1e-7)
        for v,dot in zip(vs,r["member_loss_dots"]):
            self.assertAlmostEqual(dot,math.fsum(x*y for x,y in zip(v,delta)),delta=1e-8)
        with self.assertRaises(ValueError):
            next_model.step_geometry_cohort_sgd(members,.125)
        pair,_=m.step_paired_geometry_sgd(a,b,.125)
        two,_=m.step_geometry_cohort_sgd([a,b],.125)
        self.assertEqual(pair.geometry_artifact_id,two.geometry_artifact_id)

    def test_common_geometry_admits_groups_and_checks_post_rounding_descent(self):
        m=self.model.with_lora(1,2,3.).with_geometry(artifact(self.model,path="transformer.h.1.attn",trainable=True))
        a=m.geometry_line_set_gradients([1],[[2]],stop_id=0)
        end,r=m.step_common_geometry_sgd([[a,a],[a],[a]],.125,target_l2=.002)
        one,ref=m.step_geometry_cohort_sgd([a],.125,target_l2=.002)
        self.assertEqual(end.geometry_artifact_id,one.geometry_artifact_id)
        self.assertEqual((r["group_sizes"],r["trainable_values"]),([2,1,1],4))
        self.assertEqual(r["group_target_tokens"],[2*a.target_tokens,a.target_tokens,a.target_tokens])
        self.assertAlmostEqual(sum(r["unit_group_weights"]),1.,places=12)
        self.assertAlmostEqual(sum(r["raw_group_weights"]),1.,places=12)
        self.assertAlmostEqual(r["predicted_min_unit_margin"],1.,places=10)
        self.assertGreater(r["applied_min_unit_margin"],.9999)
        for d in r["group_loss_dots"]:
            self.assertAlmostEqual(d,ref["member_loss_dots"][0],places=12)
        self.assertEqual(end.lora_to_json(),m.lora_to_json())
        with self.assertRaises(ValueError):
            end.step_common_geometry_sgd([[a],[a]],.1)
        class Unread:
            def __init__(self,n):self.n=n
            def __len__(self):return self.n
            def __getitem__(self,i):raise AssertionError("extracted items before domain admission")
        for groups,rate,dist in ((Unread(9),.1,None),([Unread(1),Unread(0)],.1,None),
                                 ([Unread(129),Unread(128)],.1,None),([Unread(1),Unread(1)],float("nan"),None),
                                 ([Unread(1),Unread(1)],.1,0.)):
            with self.assertRaises(ValueError):
                m.step_common_geometry_sgd(groups,rate,target_l2=dist)

    def test_geometry_guard_keeps_safe_bytes_and_admits_sizes_before_extraction(self):
        m = self.model.with_geometry(artifact(self.model, path="transformer.h.1.attn", trainable=True))
        g = m.geometry_line_set_gradients([1], [[2]], stop_id=0)
        proposal, _ = m.step_geometry_cohort_sgd([g], .125, target_l2=.002)
        end, r = m.guard_geometry_step(proposal, [[g]] * 9)
        self.assertEqual(end.geometry_artifact_id, proposal.geometry_artifact_id)
        self.assertEqual(r["candidate_loss_dots"], r["applied_loss_dots"])
        self.assertTrue(all(d < 0 for d in r["applied_loss_dots"]))
        self.assertEqual((r["mix"], r["feasible_mix"], r["group_sizes"]), (0., None, [1] * 9))
        self.assertEqual(r["pre_geometry_id"], m.geometry_artifact_id)
        class Unread:
            def __init__(self, n): self.n = n
            def __len__(self): return self.n
            def __getitem__(self, i): raise AssertionError("extracted items before guard admission")
        for groups, threshold in ((Unread(257), .1), ([Unread(200), Unread(57)], .1),
                                  ([Unread(0)], .1), ([Unread(1)], float("nan")), ([Unread(1)], True)):
            with self.assertRaises(ValueError):
                m.guard_geometry_step(proposal, groups, min_cosine=threshold)
        stale = end.geometry_line_set_gradients([1], [[2]], stop_id=0)
        with self.assertRaises(ValueError):
            m.guard_geometry_step(proposal, [[stale]])

    def test_guard_source_stub_matches_native_signature(self):
        import ast
        import inspect
        from pathlib import Path
        stub = Path(__file__).resolve().parents[1] / "spiraltorch" / "__init__.pyi"
        model, = [c for c in ast.parse(stub.read_text()).body if isinstance(c, ast.ClassDef) and c.name == "NativeCausalLm"]
        guard, = [m for m in model.body if isinstance(m, ast.FunctionDef) and m.name == "guard_geometry_step"]
        expected = [(a.arg, kind) for kind, args in ((inspect.Parameter.POSITIONAL_ONLY, guard.args.posonlyargs),
                    (inspect.Parameter.POSITIONAL_OR_KEYWORD, guard.args.args), (inspect.Parameter.KEYWORD_ONLY, guard.args.kwonlyargs))
                    for a in args]
        signature = inspect.signature(st.NativeCausalLm.guard_geometry_step)
        self.assertEqual(expected, [(a.name, a.kind) for a in signature.parameters.values()])
        self.assertEqual(ast.literal_eval(guard.args.kw_defaults[0]), signature.parameters["min_cosine"].default)

    def test_primary_steering_uses_real_word_and_protects_baseline(self):
        m=self.model.with_lora(1,2,3.).with_geometry(artifact(self.model,path="transformer.h.1.attn",trainable=True))
        a=m.geometry_line_set_gradients([1,3],[[2]],stop_id=0)
        b=m.geometry_context_token_preference_gradients([1,3],[4,1,3,5],6)
        anchor,_=m.step_geometry_cohort_sgd([a,a,b],.125,target_l2=.002)
        end,r=m.steer_geometry_step(anchor,[a],[[a,a]],min_cosine=1e-5)
        self.assertEqual((r["group_sizes"],r["group_target_tokens"]),([1,2],[a.target_tokens,2*a.target_tokens]))
        self.assertEqual((r["mix"],r["limiting_protected_group"]),(1.,None))
        self.assertLess(r["applied_loss_dots"][0],r["anchor_loss_dots"][0])
        self.assertEqual(r["post_geometry_id"],end.geometry_artifact_id)
        self.assertEqual(end.lora_to_json(),m.lora_to_json())
        with self.assertRaises(ValueError):
            end.steer_geometry_step(anchor,[a],[[a]])
        class Unread:
            def __init__(self,n): self.n=n
            def __len__(self): return self.n
            def __getitem__(self,i): raise AssertionError("extracted gradient before size admission")
        for primary,protected,floor in ((Unread(256),[Unread(1)],.1),(Unread(1),[Unread(0)],.1),
                                        (Unread(0),[Unread(1)],.1),(Unread(1),Unread(256),.1),
                                        (Unread(1),[],.1),(Unread(1),[Unread(1)],float("nan"))):
            with self.assertRaises(ValueError):
                m.steer_geometry_step(anchor,primary,protected,min_cosine=floor)

    def test_geometry_cohort_bounds_before_extracting_foreign_handles(self):
        m=self.model.with_geometry(artifact(self.model,trainable=True))
        class Oversize:
            def __len__(self): return 257
            def __getitem__(self,i): raise AssertionError("extracted a handle before checking the cohort bound")
        for invalid in ([],Oversize()):
            with self.assertRaises(ValueError):
                m.step_geometry_cohort_sgd(invalid,.1)
        class Unread(Oversize):
            def __len__(self): return 1
        for rate,dist in ((0.,None),(.1,-1.),(float("nan"),None)):
            with self.assertRaises(ValueError):
                m.step_geometry_cohort_sgd(Unread(),rate,target_l2=dist)
        with self.assertRaises(TypeError):
            m.step_geometry_cohort_sgd([object()],.1)

    def test_invalid_alias_indices_fail_before_prompt_token_copy(self):
        class Unread:
            def __len__(self): return 2
            def __getitem__(self,i): raise AssertionError("copied token before admitting subset")
        for ids in ([], [0,1,2], [2,0], [0,0], [3], [False], [-1]):
            with self.assertRaises((ValueError, TypeError, OverflowError)):
                self.model.line_subset_loss(Unread(),[Unread(),Unread(),Unread()],accepted_indices=ids,stop_id=0)
        with self.assertRaises(ValueError):
            self.model.geometry_line_subset_gradients([1],[[2],[3]],accepted_indices=[0],stop_id=0)

    def test_context_preference_signed_word_update_and_owners(self):
        base=self.model.with_lora(1,2,3.)
        m=base.with_geometry(artifact(base,path="transformer.h.1.attn",trainable=True))
        a,b,k=[1,3,2],[4,1,3,5,2],6
        r=m.context_token_preference(a,b,k,margin=.3)
        g=m.geometry_context_token_preference_gradients(a,b,k,margin=.3)
        self.assertEqual(g.context_token_preference_report(),r)
        self.assertIsNone(g.line_set_report())
        self.assertEqual((g.target_tokens,g.window_shape,g.geometry_artifact_id),(2,[2,6],m.geometry_artifact_id))
        gap=m.next_token_score(b,k)["nll"]-m.next_token_score(a,k)["nll"]
        self.assertAlmostEqual(r["log_probability_gap"],gap,delta=2e-6)
        self.assertAlmostEqual(r["loss"],math.log1p(math.exp(.3-gap)),delta=2e-6)
        self.assertAlmostEqual(m.context_token_preference(b,a,k)["log_probability_gap"],-gap,delta=2e-6)
        end,_=m.step_geometry_cohort_sgd([g],.035,target_l2=.002)
        self.assertGreater(end.context_token_preference(a,b,k)["log_probability_gap"],gap)
        self.assertEqual(end.lora_to_json(),m.lora_to_json())
        with self.assertRaises(ValueError):
            end.step_geometry_cohort_sgd([g],.035)

    def test_context_preference_admission_before_token_transport(self):
        class Unread:
            def __init__(self,n):self.n=n
            def __len__(self):return self.n
            def __getitem__(self,i):raise AssertionError("copied tokens before admission")
        for a,b,k,m in ((0,1,2,None),(1,10000,2,None),(1,1,11,None),(1,1,2,-1.),(1,1,2,float("nan"))):
            with self.assertRaises(ValueError):
                self.model.context_token_preference(Unread(a),Unread(b),k,margin=m)
        with self.assertRaises((ValueError,TypeError)):
            self.model.context_token_preference(Unread(1),Unread(1),2,margin=True)
        with self.assertRaises(ValueError):
            self.model.geometry_context_token_preference_gradients(Unread(1),Unread(1),2)
        with self.assertRaises(ValueError):
            self.model.context_token_preference([1],[1],2)

    def test_label_free_scores_match_independent_tokens_including_interior_stops(self):
        p, menu = [1,3], [[2],[2,4],[2,4,7],[2,5]]
        r = self.model.score_lines(p, menu, stop_id=0)
        self.assertEqual((r["stop_id"],r["target_counts"],r["prefix_nodes"],r["decoder_lanes"],r["shared_content_tokens"]),
                         (0,[2,3,4,3],5,2,1))
        self.assertTrue(set(r).isdisjoint({"gold_index","accepted_indices","loss","mix"}))
        for i,line in enumerate(menu):
            prefix = p.copy()
            self.assertEqual(len(r["token_nlls"][i]),len(line)+1)
            for k,cost in zip([*line,0],r["token_nlls"][i]):
                self.assertAlmostEqual(cost,self.model.next_token_score(prefix,k)["nll"],delta=2e-6)
                prefix.append(k)
            old = self.model.line_menu_loss(p,menu,gold_index=i,stop_id=0,mix=1.)
            self.assertEqual(r["line_nlls"],old["line_nlls"])
            self.assertEqual(r["conditional_nlls"][i],old["menu_nll"])
            self.assertEqual(r["winner_index"],old["winner_index"])
            self.assertAlmostEqual(math.fsum(r["token_nlls"][i]),r["line_nlls"][i],delta=1e-12)
        self.assertEqual(r["menu_set_nll"],self.model.line_set_loss(p,menu,stop_id=0)["set_nll"])
        single = self.model.score_lines(p, [menu[0]], stop_id=0)
        self.assertEqual(single["conditional_nlls"],[0.])
        self.assertAlmostEqual(single["line_nlls"][0],single["menu_set_nll"],delta=1e-12)

    def test_label_free_score_bounds_precede_token_copy(self):
        class Unread:
            def __init__(self, n): self.n=n
            def __len__(self): return self.n
            def __getitem__(self, _): raise AssertionError("copied an inadmissible token")
        for menu in (Unread(33), [Unread(4096)], []):
            with self.assertRaises(ValueError):
                self.model.score_lines(Unread(1),menu,stop_id=0)
        for menu in ([[2],[2]], [[3],[2]], [[]], [[11]], [[0]]):
            with self.assertRaises(ValueError):
                self.model.score_lines([1],menu,stop_id=0)
        with self.assertRaises((ValueError,TypeError)):
            self.model.score_lines([1],[[2]],stop_id=True)

    def test_line_set_admission_and_forward_without_trainable_geometry(self):
        class Unread:
            def __len__(self):
                return 70000
            def __getitem__(self, index):
                raise AssertionError("inadmissible values were copied")
        self.assertGreater(self.model.line_set_loss([1], [[2]], stop_id=0)["set_nll"], 0.)
        for candidates in ([], [[]], [[1], [1]], [[2], [1]], [[0]], [[11]], [Unread()]):
            with self.assertRaises(ValueError):
                self.model.line_set_loss([1], candidates, stop_id=0)
        with self.assertRaises((TypeError, ValueError)):
            self.model.line_set_loss([1], [[2]], stop_id=False)
        with self.assertRaises(ValueError):
            self.model.geometry_line_set_gradients(Unread(), [[2]], stop_id=0)

    def test_line_set_admits_prefix_sharing_before_readout_but_not_unbounded_ids(self):
        class Unread:
            def __len__(self):
                return 1
            def __getitem__(self, index):
                raise AssertionError("read tokens despite an impossible minimum prefix count")
        def limited(rows):
            return st.NativeCausalLm.from_gpt2(self.config, self.weights, max_logit_elements=rows*11)
        prompt, lines = [1,3], [[2], [2,4], [2,4,7], [2,5]]
        model = limited(5)
        self.assertEqual(model.line_set_loss(prompt, lines, stop_id=0)["decoder_lanes"], 2)
        with self.assertRaises(ValueError):
            model.line_set_loss(prompt, [[2],[3,4],[3,5,6],[8,9]], stop_id=0)
        with self.assertRaises(ValueError):
            limited(4).line_set_loss(Unread(), lines, stop_id=0)
        r = model.line_menu_loss(prompt, lines, gold_index=1, stop_id=0, mix=1.)
        self.assertEqual((r["decoder_lanes"], r["prefix_nodes"]), (2, 5))
        with self.assertRaises(ValueError):
            model.line_menu_loss(prompt, [[2],[3,4],[3,5,6],[8,9]], gold_index=1, stop_id=0, mix=1.)

    def test_complete_line_admission_rejects_bad_support_before_token_copy(self):
        class Unread:
            def __init__(self, length):
                self.length = length
            def __len__(self):
                return self.length
            def __getitem__(self, index):
                raise AssertionError("copied a token before admitting work")
        m = self.model.with_geometry(artifact(self.model,trainable=True))
        for prompt, cs, gold, stop, mix in [
                (Unread(0),[Unread(1),Unread(1)],0,0,1.), (Unread(8),[Unread(1),Unread(1)],0,0,1.),
                (Unread(1),Unread(33),0,0,1.), (Unread(1),[Unread(1),Unread(1)],2,0,1.),
                (Unread(1),[Unread(1),Unread(30000)],0,0,1.), (Unread(1),[Unread(1),Unread(1)],0,11,1.),
                (Unread(1),[Unread(1),Unread(1)],0,0,float("nan")), (Unread(1),[Unread(1),Unread(1)],0,0,True)]:
            with self.assertRaises((ValueError,TypeError)):
                m.geometry_line_menu_gradients(prompt,cs,gold_index=gold,stop_id=stop,mix=mix)
        with self.assertRaises(ValueError):
            self.model.geometry_line_menu_gradients(Unread(1),Unread(2),gold_index=0,stop_id=0,mix=1.)
        for menu in ([[2],[2]], [[3],[2]], [[0],[2]], [[1],[11]]):
            with self.assertRaises(ValueError):
                self.model.line_menu_loss([1],menu,gold_index=0,stop_id=0,mix=1.)

    def test_anchored_projection_is_opt_in_and_training_reaches_decoder(self):
        path = "transformer.h.1.attn"
        state = json.loads(artifact(self.model, path=path, trainable=True).state_json())
        placement = state["placements"][0]
        placement["recipe"] = dict(schema="spiraltorch.wave_stage_adapter.v1", features=4, strength=.5,
                                   max_values=1024, stage=dict(kind="projection", curvature=-1., log_radius=0.))
        original = st.GeometryAdapterArtifact.create_json(json.dumps(state))
        anchored = original.anchor_wave_projection(path)
        base = self.model.with_lora(1, 2, 3.).with_readout_accumulator("f32_wide_backward")
        ordinary, centered = base.with_geometry(original), base.with_geometry(anchored)
        ids = [1, 5, 2, 6]
        def logits(m):
            return bytes(m.forward(ids, [1, 4], last_only=True).values_buffer())
        def gradient(m):
            return m.adaptation_loss_gradients(ids, [1, 4], mode="geometry")
        g, h = gradient(ordinary), gradient(centered)
        self.assertEqual(logits(ordinary), logits(centered))
        self.assertEqual(g.parameter_gradients(), h.parameter_gradients())
        with self.assertRaises(ValueError):
            centered.step_paired_geometry_sgd(g, h, .5)
        new, receipt = centered.step_paired_geometry_sgd(h, h, .5, policy="common_descent")
        self.assertEqual(receipt["trainable_values"], 4)
        self.assertEqual(new.lora_to_json(), ordinary.lora_to_json())
        out = new.geometry_artifact()
        restored = st.GeometryAdapterArtifact.from_json(out.to_json(), out.artifact_id, new.base_model_identity)
        self.assertEqual(logits(new), logits(base.with_geometry(restored)))
        self.assertNotEqual(logits(new), logits(centered))

    def test_paired_geometry_keeps_equal_descent_exact_and_rejects_foreign_owners(self):
        model = self.model.with_lora(1, 2, 3.).with_geometry(
            artifact(self.model, path="transformer.h.1.attn", trainable=True)
        ).with_readout_accumulator("f32_wide_backward")
        ids = [1, 2, 3, 5]
        def grads(m, mode):
            return m.adaptation_loss_gradients(ids, [1, len(ids)], mode=mode)
        g = grads(model, "geometry")
        ordinary = model.step_adaptation_sgd(g, "geometry", .75)
        eq, a = model.step_paired_geometry_sgd(g, g, .75)
        guarded, b = model.step_paired_geometry_sgd(
            g, g, .75, policy="common_descent", target_l2=a["applied"]["l2"])
        self.assertEqual(ordinary.geometry_artifact_id, eq.geometry_artifact_id)
        self.assertEqual(guarded.geometry_artifact_id, eq.geometry_artifact_id)
        self.assertEqual(guarded.lora_to_json(), model.lora_to_json())
        self.assertNotEqual(eq.geometry_artifact_id, model.geometry_artifact_id)
        self.assertEqual(b["pre_geometry_id"], model.geometry_artifact_id)
        self.assertEqual(b["post_geometry_id"], guarded.geometry_artifact_id)
        self.assertEqual(b["readout_accumulator"], "f32_wide_backward")
        self.assertEqual(b["trainable_values"], 4)
        self.assertEqual(b["weights"], [.5, .5])
        self.assertFalse(b["reweighted"])
        self.assertTrue(b["applied"]["both_descend"])
        for other in (grads(ordinary, "geometry"), grads(model.with_readout_accumulator("f32"), "geometry"),
                      grads(model, "joint"), grads(model, "lora")):
            for x, y in ((other, g), (g, other)):
                with self.assertRaises(ValueError):
                    model.step_paired_geometry_sgd(x, y, .75)
        for rate, options in ((True, {}), (0., {}), (float("nan"), {}), (1., {"policy":"unknown"}),
                              (1., {"target_l2":True}), (1., {"target_l2":-1.})):
            with self.assertRaises((ValueError, TypeError)):
                model.step_paired_geometry_sgd(g, g, rate, **options)

    def test_distance_control_admits_owners_and_work_before_foreign_token_reads(self):
        class Unread:
            def __len__(self):
                raise AssertionError("read forbidden token collection")
        base = self.model
        geo = artifact(base, path="transformer.h.1.attn", trainable=True)
        ce = base.with_geometry(geo)
        params = dict(learning_rate=.75, rivals=2, mix=.5)
        for mixed, shape, changes in [(base, [1,3], {}), (ce.with_readout_accumulator("f64"), [1,3], {}),
                    (base.with_lora(1,2,3.).with_geometry(geo), [1,3], {}), (ce, [1,12], {}),
                    (ce, [1,3], {"learning_rate":0.}), (ce, [1,3], {"learning_rate":True}),
                    (ce, [1,3], {"rivals":11}), (ce, [1,3], {"mix":True})]:
            with self.assertRaises((ValueError, TypeError)):
                ce.geometry_competition_pair_step(mixed, Unread(), shape, **{**params, **changes})
        grad = ce.geometry_competition_gradients([1,2,3], [1,3], rivals=2, mix=.5)
        for lr, distance in [(-1.,0.), (True,0.), (.75,False), (.75,-1.), (.75,float("nan"))]:
            with self.assertRaises((ValueError, TypeError)):
                ce.step_geometry_sgd_at_distance(grad, lr, distance)
        updated = ce.step_adaptation_sgd(grad, "geometry", .75)
        with self.assertRaises(ValueError):
            updated.step_geometry_sgd_at_distance(grad, .75, 0.)

    def test_coverage_guard_is_opt_in_and_uses_reference_members(self):
        reference = self.model.with_geometry(artifact(self.model, path="transformer.h.1.attn", trainable=True))
        other = self.model.with_geometry(artifact(self.model, path="transformer.h.1.attn", gate=-.8, trainable=True))
        ref = reference.forward([2,5],[1,2],last_only=True).values()
        cand = other.forward([2,5],[1,2],last_only=True).values()
        def cost(row, subset):
            peak=max(row)
            return math.log1p(sum(math.exp(row[i]-peak) for i in range(11) if i not in subset) /
                              sum(math.exp(row[i]-peak) for i in subset))
        active=0
        for target in range(11):
            ids=[2,5,target];kw=dict(learning_rate=.75,rivals=2,mix=.5,labels=[-100,-100,target])
            c0,m0,a=reference.geometry_competition_pair_step(other,ids,[1,3],**kw)
            _,z,zr=reference.geometry_competition_pair_step(other,ids,[1,3],coverage_guard=0.,**kw)
            self.assertEqual((m0.geometry_artifact_id,a),(z.geometry_artifact_id,zr))
            c1,m1,b=reference.geometry_competition_pair_step(other,ids,[1,3],coverage_guard=.5,**kw)
            members=[target,*sorted((i for i in range(11) if i!=target),key=lambda i:(-ref[i],i))[:2]]
            gap=cost(cand,members)-cost(ref,members)
            guard=b["competition"]["coverage_guard"]
            self.assertEqual(guard["active_targets"],int(gap>0))
            self.assertAlmostEqual(guard["mean_violation"],max(0.,gap),places=13)
            self.assertEqual(c0.geometry_artifact_id,c1.geometry_artifact_id)
            if gap<=0:self.assertEqual(m1.geometry_artifact_id,m0.geometry_artifact_id)
            active+=guard["active_targets"]
        self.assertTrue(0<active<11)
        for bad in (True, -1., 1.1, float("nan")):
            with self.assertRaises((ValueError,TypeError)):
                reference.geometry_competition_pair_step(other,[],[1,3],coverage_guard=bad,**kw)

    def test_rival_projection_update_roundtrips_and_replays_f32_sgd(self):
        state = json.loads(artifact(self.model, path="transformer.h.0.attn", trainable=True).state_json())
        site = state["placements"][0]
        site["recipe"] = dict(schema="spiraltorch.wave_stage_adapter.v1", features=4, strength=.25, max_values=1024,
                              stage=dict(kind="projection", curvature=-1., log_radius=0.))
        for t in site["tensors"]:
            t["requires_grad"] = True
        model = self.model.with_lora(0, 2, 3.).with_geometry(st.GeometryAdapterArtifact.create_json(json.dumps(state)))
        g = model.geometry_competition_gradients([1, 2, 3, 4], [1, 4], labels=[-100]*3+[4], rivals=3, mix=.5)
        nxt = model.step_adaptation_sgd(g, "geometry", .01)
        saved = nxt.geometry_artifact()
        loaded = st.GeometryAdapterArtifact.from_json(saved.to_json(), saved.artifact_id, model.base_model_identity)
        self.assertEqual(model.with_geometry(loaded).geometry_artifact_id, nxt.geometry_artifact_id)
        params = g.parameter_gradients()[site["path"]]
        f32 = lambda v: struct.unpack("<f", struct.pack("<f", v))[0]
        after = {t["name"]: t["values"] for t in json.loads(loaded.state_json())["placements"][0]["tensors"]}
        for t in site["tensors"]:
            vals = [f32(f32(x) - f32(f32(.01)*dx)) for x, dx in zip(t["values"], params[t["name"]])]
            self.assertEqual(vals, [f32(x) for x in after[t["name"]]])
        self.assertEqual(model.lora_to_json(), nxt.lora_to_json())

    def test_global_geometry_displacement_preserves_frozen_owners(self):
        origin = artifact(self.model, path="transformer.h.0.attn", gate=0., trainable=True)
        state = json.loads(origin.state_json())
        gate = next(t for t in state["placements"][0]["tensors"] if t["name"] == "gate")
        gate["values"] = [3., -4., 0., 0.]
        end = st.GeometryAdapterArtifact.create_json(json.dumps(state))
        self.assertEqual(origin.trainable_delta_l2(end), 5.)
        half, info = origin.match_trainable_delta(end, target_l2=2.5)
        self.assertEqual((info["direction_l2"], info["realized_l2"], info["trainable_values"],
                          info["relative_vector_error"]), (5., 2.5, 4, 0.))
        self.assertEqual(info["result_id"], half.artifact_id)
        self.assertEqual(origin.match_trainable_delta(end, 0.)[0].artifact_id, origin.artifact_id)
        self.assertEqual(origin.match_trainable_delta(end, 5.)[0].artifact_id, end.artifact_id)
        model = self.model.with_lora(0, 2, 3.).with_geometry(origin)
        changed = model.with_geometry(half)
        self.assertEqual(changed.lora_to_json(), model.lora_to_json())
        self.assertEqual(changed.base_model_identity, model.base_model_identity)
        self.assertEqual(changed.geometry_artifact().to_json(), half.to_json())
        for invalid in [True, -1., float("nan"), float("inf")]:
            with self.assertRaises((ValueError,TypeError)):
                origin.match_trainable_delta(end, invalid)
        with self.assertRaises(ValueError):
            origin.match_trainable_delta(origin, 1.)

    def test_rejects_invalid_ids_shapes_and_unsupported_configuration(self):
        for ids, shape in [([11], [1, 1]), ([-1], [1, 1]), ([True], [1, 1]), ([], [1, 1]),
                           ([1], [True, 1]), ([], [1, 9]), ([], [0, 1])]:
            with self.subTest(shape=shape, ids=ids), self.assertRaises((ValueError, TypeError, OverflowError)):
                self.model.forward(ids, shape)
        for key, value in (("model_type", "llama"), ("activation_function", "relu"),
                           ("add_cross_attention", True), ("reorder_and_upcast_attn", True)):
            config = json.loads(self.config)
            config[key] = value
            with self.assertRaises(ValueError):
                st.NativeCausalLm.from_gpt2(json.dumps(config), self.weights)
        with self.assertRaises(TypeError):
            st.NativeCausalLm.from_gpt2(self.config, self.weights, max_logit_elements=True)

    def test_work_budget_precedes_python_token_extraction(self):
        class Poison:
            def __len__(self):
                raise AssertionError("tokens were read before work admission")
        small = st.NativeCausalLm.from_gpt2(self.config, self.weights, max_logit_elements=11)
        with self.assertRaisesRegex(ValueError, "logits"):
            small.forward(Poison(), [1, 2])
        with self.assertRaisesRegex(ValueError, "logits"):
            small.next_token_loss(Poison(), [1, 3])
        self.assertEqual(small.forward([1, 2], [1, 2], last_only=True).shape, [1, 1, 11])
        limited = st.NativeCausalLm.from_gpt2(self.config, self.weights, max_attention_pairs=15)
        with self.assertRaisesRegex(ValueError, "attention pairs"):
            limited.forward(Poison(), [1, 2], last_only=True)

    def test_native_language_loss_learning_keeps_base_and_resumes_exactly(self):
        base = self.model
        state = artifact(base, trainable=True)
        original = base.with_geometry(state)
        windows, shape = [1, 2, 3, 4, 5, 6, 7, 8], [2, 4]
        first = original.geometry_loss_gradients(windows, shape)
        self.assertEqual(first.loss, original.next_token_loss(windows, shape))
        self.assertEqual(first.window_shape, shape)
        self.assertEqual(first.base_model_identity, base.base_model_identity)
        self.assertEqual(first.geometry_artifact_id, state.artifact_id)
        gradients = first.parameter_gradients()
        self.assertTrue(any(x != 0. for x in gradients["transformer.h.0.mlp"]["gate"]))
        gradients["transformer.h.0.mlp"]["gate"][0] = 999.
        self.assertNotEqual(first.parameter_gradients(), gradients)
        model = original.step_geometry_sgd(first, .5)
        with self.assertRaisesRegex(ValueError, "stale"):
            model.step_geometry_sgd(first, .5)
        for _ in range(19):
            model = model.step_geometry_sgd(model.geometry_loss_gradients(windows, shape), .5)
        self.assertLess(model.next_token_loss(windows, shape), first.loss)
        self.assertEqual(model.base_model_identity, base.base_model_identity)
        self.assertIsNone(base.geometry_artifact())
        self.assertEqual(original.geometry_artifact().to_json(), state.to_json())
        saved = model.geometry_artifact()
        restored_state = st.GeometryAdapterArtifact.from_json(saved.to_json(), saved.artifact_id, base.base_model_identity)
        restored = base.with_geometry(restored_state)
        a, b = model.geometry_loss_gradients(windows, shape), restored.geometry_loss_gradients(windows, shape)
        self.assertEqual(a.parameter_gradients(), b.parameter_gradients())
        self.assertEqual(model.step_geometry_sgd(a, .5).geometry_artifact().to_json(),
                         restored.step_geometry_sgd(b, .5).geometry_artifact().to_json())
        before = json.loads(state.state_json())["placements"][0]["tensors"]
        after = json.loads(saved.state_json())["placements"][0]["tensors"]
        self.assertEqual(next(t for t in before if t["name"] == "bias"), next(t for t in after if t["name"] == "bias"))
        self.assertIn("NativeCausalLmGradients", st.__all__)
        with self.assertRaises(TypeError):
            st.NativeCausalLmGradients()

    def test_learning_budget_and_flags_precede_token_extraction(self):
        class Poison:
            def __len__(self):
                raise AssertionError("token extraction must follow learning admission")
        with self.assertRaisesRegex(ValueError, "no attached geometry"):
            self.model.geometry_loss_gradients(Poison(), [1, 3])
        frozen = self.model.with_geometry(artifact(self.model))
        with self.assertRaisesRegex(ValueError, "no trainable parameters"):
            frozen.geometry_loss_gradients(Poison(), [1, 3])
        small = st.NativeCausalLm.from_gpt2(self.config, self.weights, max_training_checkpoint_elements=23)
        small = small.with_geometry(artifact(small, trainable=True))
        with self.assertRaisesRegex(ValueError, "training checkpoints"):
            small.geometry_loss_gradients(Poison(), [1, 3])
        g = small.geometry_loss_gradients([1, 2], [1, 2])
        before = small.geometry_artifact().to_json()
        for rate in (True, 0., -1., float("inf"), float("nan")):
            with self.assertRaises((ValueError, TypeError)):
                small.step_geometry_sgd(g, rate)
            self.assertEqual(small.geometry_artifact().to_json(), before)


if __name__ == "__main__":
    unittest.main()
