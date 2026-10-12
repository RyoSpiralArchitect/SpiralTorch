"""Fabricated protocol counterexamples, not corpus-learning evidence."""
import copy
import unittest

import byte_corpus_study as study
import byte_corpus_headwise as headwise
import byte_corpus_calibration as calibration
from test_byte_corpus_fisher import fixture as fisher_fixture
import verify_byte_corpus_calibrated_controls as controls
import verify_byte_corpus_headwise_preparation as preparation


def fixture():
    raw, reference, actual = fisher_fixture()
    source = study.decode_json(raw)
    source["config"]["heads"] = 2
    for case in source["cases"]:
        if case["geometry"]:
            case["parameters"][6]["shape"] = [2]
            case["parameters"][6]["values"] *= 2
    for report in (reference, actual):
        for row, spec in zip(report["cases"], source["cases"]):
            if row["geometry"]:
                row["final_parameters"][6] *= 2
            row["parameter_scalars"] = sum(len(p["values"]) for p in spec["parameters"])
            row["trainable_parameter_scalars"] = study.trainable_scalars(spec)
    for row in reference["calibration_verification"]["cases"]:
        for key in ("reference_rms", "fitted_rms", "realized_relative_errors"):
            row[key][0] *= 2
    source_raw = study.encoded(source)
    for report in (reference, actual):
        report["request_sha256"] = study.digest(source_raw)
    retained = copy.deepcopy(actual)
    request = copy.deepcopy(source)
    request["schema"] = study.REQUEST_V6
    request["headwise_controls"] = {"source_request_sha256":study.digest(source_raw)}
    bases = {m:next(c for c in source["cases"] if c.get("pair_metric") == m
              and c["geometry_update"] == "train" and study.matched(c) == (m != study.POINCARE))
             for m in (study.POINCARE, study.FLAT, study.FISHER)}
    for route in headwise.ROUTES:
        gain = [bases[m]["parameters"][6]["values"][h] for h,m in enumerate(route)]
        for update in ("train", "frozen"):
            case = copy.deepcopy(bases[study.POINCARE])
            case.update(name=f"{route}-{update}", head_pair_metrics=[list(route)],
                        geometry_update=update, bias_initialization=study.MATCHED)
            del case["pair_metric"]
            case["parameters"][6]["values"] = gain[:]
            request["cases"].append(case)
            for report in (reference, actual):
                row = copy.deepcopy(report["cases"][1 if update == "train" else 2])
                row.update(name=case["name"], head_pair_metrics=case["head_pair_metrics"],
                           geometry_update=update, bias_initialization=study.MATCHED)
                del row["pair_metric"]
                row["trainable_parameter_scalars"] = study.trainable_scalars(case)
                row["final_parameters"][6] = [study.f32(v + (.01 if update == "train" else 0)) for v in gain]
                report["cases"].append(row)
        qualification = copy.deepcopy(reference["calibration_verification"]["cases"][0])
        qualification.update(name=request["cases"][-2]["name"], head_pair_metrics=[list(route)])
        del qualification["pair_metric"]
        reference["calibration_verification"]["cases"].append(qualification)
    raw = study.encoded(request)
    for report in (reference, actual):
        report.update(schema="spiraltorch.byte_corpus.result.v6", request_sha256=study.digest(raw))
    return raw, reference, actual, source_raw, retained


def packet_fixture():
    raw, ref, _, source, _ = fixture()
    request = study.decode_json(raw)
    records = []
    for case, measured in zip(request["cases"][9::2], ref["calibration_verification"]["cases"][2:]):
        p = case["parameters"]
        linear = lambda i, gelu=False: dict(kind="linear",weight=i,bias=i+1,gelu=gelu)
        norm = dict(kind="layer_norm",gain=0,bias=1,epsilon=1e-5)
        def graph(parameters, stages, roles):
            return dict(schema=f"spiraltorch.nn.inference_plan.v{3 if stages[0]['kind'] == 'layer_norm' else 2}",
                        input_shape=[request["config"][k] for k in ("batch","steps","width")],
                        parameters=[dict(shape=x["shape"], values=x["values"],role=role) for x,role in zip(parameters,roles)], stages=stages)
        model = dict(schema="spiraltorch.nn.byte_decoder_checkpoint.v3", update_rule="stateless_sgd.v1",
            window_state="reset_positions_and_geometry.v1", attempted_revision="0", model=dict(
                token=dict(shape=p[0]["shape"], values=p[0]["values"]),
                position=dict(shape=p[1]["shape"], values=p[1]["values"]),
                geometry=dict(projection=graph(p[2:4],[linear(0)],["weight","bias"]), curvature=request["config"]["curvature"],
                              raw_decay=p[4]["values"], raw_phase=p[5]["values"], raw_gains=[p[6]["values"]],
                              head_pair_metrics=case["head_pair_metrics"]),
                blocks=[dict(pre=graph(p[7:9],[norm],["gain","bias"]),heads=2,
                             qkv=graph(p[9:11],[linear(0)],["weight","bias"]),output=graph(p[11:13],[linear(0)],["weight","bias"]),
                             feed_forward=graph(p[13:-4],[norm,linear(2,True),linear(4)],["gain","bias","weight","bias","weight","bias"]))],
                head=graph(p[-4:],[norm,linear(2)],["gain","bias","weight","bias"])))
        records.append(dict(name=case["name"],seed=case["seed"],head_pair_metrics=case["head_pair_metrics"],
            raw_gains=p[6]["values"], target_rms=measured["reference_rms"], actual_rms=measured["fitted_rms"],
            relative_errors=measured["realized_relative_errors"],valid_pairs_per_head=[measured["valid_pairs_per_head"]],
            model_json=study.encoded(model).decode()))
    report = dict(schema="spiraltorch.byte_corpus.headwise_preparation.v1",source_request_sha256=study.digest(source),
        request_sha256=study.digest(raw), adapter="fabricated test data", passed=True, no_fitting=True,
        no_updates_consumed=True, train_batch_indices=request["bias_calibration"]["train_batch_indices"],
        relative_tolerance=1e-5, cases=records)
    return source, dict(schema="spiraltorch.byte_corpus.prepared.v1",request_json=raw.decode(),
                        report_json=study.encoded(report).decode())


class HeadwiseStudy(unittest.TestCase):
    def test_preparation_binds_source_grid_weights_counts_and_rms(self):
        source, packet = packet_fixture()
        self.assertTrue(preparation.verify(source, packet)["passed"])
        for mutation in ("missing", "route", "gain", "checkpoint", "count", "rms", "source"):
            bad = copy.deepcopy(packet)
            report = study.decode_json(bad["report_json"])
            if mutation == "missing": report["cases"].pop()
            if mutation == "route": report["cases"][0]["head_pair_metrics"][0].reverse()
            if mutation == "gain": report["cases"][0]["raw_gains"][0] += .01
            if mutation == "checkpoint":
                model = study.decode_json(report["cases"][0]["model_json"])
                model["model"]["geometry"]["head_pair_metrics"][0].reverse()
                report["cases"][0]["model_json"] = study.encoded(model).decode()
            if mutation == "count": report["cases"][0]["valid_pairs_per_head"][0] += 1
            if mutation == "rms": report["cases"][0]["actual_rms"][0][1] *= 1.01
            if mutation == "source": report["source_request_sha256"] = "b" * 64
            bad["report_json"] = study.encoded(report).decode()
            with self.assertRaises(ValueError):
                preparation.verify(source, bad)

    def test_preparation_rejects_modified_topology(self):
        source, packet = packet_fixture()
        for mutation in ("heads","curvature","epsilon","stages","layout","role","extra"):
            bad = copy.deepcopy(packet)
            report = study.decode_json(bad["report_json"])
            checkpoint = study.decode_json(report["cases"][0]["model_json"])
            model = checkpoint["model"]
            if mutation == "heads": model["blocks"][0]["heads"] = 1
            if mutation == "curvature": model["geometry"]["curvature"] = -.5
            if mutation == "epsilon": model["blocks"][0]["pre"]["stages"][0]["epsilon"] = .1
            if mutation == "stages": model["geometry"]["projection"]["stages"] = []
            if mutation == "layout": model["head"]["input_shape"][1] += 1
            if mutation == "role": model["head"]["parameters"][0]["role"] = "bias"
            if mutation == "extra": model["blocks"][0]["pre"]["stages"][0]["unused"] = 1
            report["cases"][0]["model_json"] = study.encoded(checkpoint).decode()
            bad["report_json"] = study.encoded(report).decode()
            with self.assertRaises(ValueError):
                preparation.verify(source, bad)

    def test_heterogeneous_parameters_reject_coercible_types(self):
        for field, bad in (("shape",[2.0]),("shape",[True,2]),("values",[True,True]),
                           ("values",[float("nan"),1.]),("values",["1",1.])):
            request = study.decode_json(fixture()[0])
            param = next(p for p in request["cases"][9]["parameters"] if p["name"] == "block.0.pre.gain")
            param[field] = bad
            with self.assertRaises(ValueError):
                study.request_version(request)

    def test_all_twenty_one_modes_and_reversal_contrasts(self):
        raw, ref, actual, _, _ = fixture()
        request = study.decode_json(raw)
        self.assertEqual(study.request_version(request), 6)
        for _ in range(21):
            request["cases"] = request["cases"][1:] + request["cases"][:1]
            self.assertEqual(study.request_version(request), 6)
        result = study.compare(raw, ref, actual)
        self.assertEqual(len(result["headwise_contrasts"]), 6)
        self.assertTrue(all(all(c["same_sign_as_torch"].values()) for c in result["headwise_contrasts"]))
        self.assertTrue(all(len(c["geometry_deltas"][-1]["per_head"]) == 2
            for c in result["cases"] if "head_pair_metrics" in c and c["geometry_update"] == "train"))

    def test_gain_inheritance_not_arbitrary_matching_twins(self):
        for index in range(9,21):
            for head in range(2):
                r = study.decode_json(fixture()[0])
                r["cases"][index]["parameters"][6]["values"][head] += .01
                with self.assertRaises(ValueError):
                    study.request_version(r)
        r = study.decode_json(fixture()[0])
        for c in r["cases"][9:11]:
            c["parameters"][6]["values"] = list(reversed(c["parameters"][6]["values"]))
        with self.assertRaises(ValueError):
            study.request_version(r)

    def test_missing_conflicting_and_legacy_routes_reject(self):
        for mutation in ("null", "duplicate", "both", "missing", "schema", "source", "metadata"):
            r = study.decode_json(fixture()[0])
            if mutation == "null": r["cases"][9]["head_pair_metrics"] = None
            if mutation == "duplicate": r["cases"][9]["head_pair_metrics"] = r["cases"][11]["head_pair_metrics"]
            if mutation == "both": r["cases"][9]["pair_metric"] = study.POINCARE
            if mutation == "missing": del r["cases"][9]["head_pair_metrics"]
            if mutation == "schema": r["schema"] = study.REQUEST_V5
            if mutation == "source": r["headwise_controls"]["source_request_sha256"] = "?"
            if mutation == "metadata": del r["headwise_controls"]
            with self.assertRaises(ValueError):
                study.request_version(r)

    def test_rms_requires_every_head_and_one_shared_target(self):
        raw, ref, _, _, _ = fixture()
        r = study.decode_json(raw)
        calibration.verify_qualification(r, ref["calibration_verification"])
        for mutation in ("missing", "route", "count", "target", "error", "nan"):
            bad = copy.deepcopy(ref["calibration_verification"])
            c = bad["cases"][2]
            if mutation == "missing": bad["cases"].pop()
            if mutation == "route": c["head_pair_metrics"][0].reverse()
            if mutation == "count": c["valid_pairs_per_head"] += 1
            if mutation == "target":
                c["reference_rms"][0][1] *= 2
                c["fitted_rms"][0][1] *= 2
            if mutation == "error": c["fitted_rms"][0][1] *= 1.01
            if mutation == "nan": c["fitted_rms"][0][1] = float("nan")
            with self.assertRaises(ValueError):
                calibration.verify_qualification(r, bad)

    def test_severed_small_head_cannot_hide_behind_larger_delta(self):
        with self.assertRaises(ValueError):
            headwise.gain_deltas([0.,0.], [10.,0.], [10.,1e-5])
        with self.assertRaises(ValueError):
            headwise.gain_deltas([0.,0.], [10.,1e-9], [10.,1e-9])
        self.assertEqual(len(headwise.gain_deltas([0.,0.], [10.,1e-5], [10.,1e-5])), 2)

    def test_inherited_nine_and_ten_first_update_pairs(self):
        raw, _, full, source, retained = fixture()
        first = copy.deepcopy(full)
        first["schema"] = "spiraltorch.byte_corpus.partial.v6"
        for c in first["cases"]:
            c["training"] = c["training"][:1]
            c["validation"] = c["validation"][:1]
        result = controls.verify(raw, source, full, retained, first)
        self.assertEqual(len(result["inherited_full_reports_exact"]), 9)
        self.assertEqual(len(result["pairs"]), 10)
        for mutation in ("route", "backbone", "frozen", "history"):
            bad = copy.deepcopy(first)
            c = bad["cases"][10]
            if mutation == "route": c["head_pair_metrics"][0].reverse()
            if mutation == "backbone": c["final_parameters"][0][0] += .01
            if mutation == "frozen": c["final_parameters"][6][1] += .01
            if mutation == "history": c["training"] = []
            with self.assertRaises(ValueError):
                controls.verify(raw, source, full, retained, bad)


if __name__ == "__main__":
    unittest.main()
