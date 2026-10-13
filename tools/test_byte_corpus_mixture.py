"""Fabricated protocol counterexamples, not corpus-learning evidence."""
import copy
import unittest

import byte_corpus_study as study
import byte_corpus_mixture as mixture
import byte_corpus_calibration as calibration
import verify_byte_corpus_calibrated_controls as controls
import verify_byte_corpus_mixture_preparation as preparation
import verify_byte_corpus_resume as resume
from test_byte_corpus_headwise import fixture as headwise_fixture, packet_fixture as headwise_packet


def fixture():
    _, head_ref, _, source_raw, retained = headwise_fixture()
    source = study.decode_json(source_raw)
    request = copy.deepcopy(source)
    request.update(schema=study.REQUEST_V7,
                   mixture_controls={"source_request_sha256":study.digest(source_raw)})
    actual = copy.deepcopy(retained)
    reference = copy.deepcopy(head_ref)
    reference["cases"] = reference["cases"][:9]
    reference["calibration_verification"]["cases"] = reference["calibration_verification"]["cases"][:2]
    bases = {m:next(c for c in source["cases"] if c.get("pair_metric")==m
             and c["geometry_update"]=="train" and study.matched(c)==(m!=study.POINCARE))
             for m in (mixture.P,mixture.E,mixture.F)}
    for metrics in mixture.COMBINATIONS:
        for update in ("train","frozen"):
            case = copy.deepcopy(bases[mixture.P])
            case.update(name=f"{metrics}-{update}", bias_initialization=study.MATCHED,
                        mixture=dict(metrics=list(metrics),scales=[mixture.unit_scales(bases,metrics)],router_update=update))
            del case["pair_metric"]
            case["parameters"].insert(7,dict(name="geometry.mixture_logits.0",shape=[2,len(metrics)],values=[0.]*(2*len(metrics))))
            request["cases"].append(case)
            for report in (reference,actual):
                row = copy.deepcopy(report["cases"][1])
                row.update(name=case["name"],bias_initialization=study.MATCHED,mixture=copy.deepcopy(case["mixture"]),
                           parameter_tensors=len(case["parameters"]),parameter_scalars=sum(len(p["values"]) for p in case["parameters"]),
                           trainable_parameter_scalars=study.trainable_scalars(case))
                del row["pair_metric"]
                delta = [-.01,.01] if len(metrics)==2 else [-.01,.02,-.01]
                row["final_parameters"].insert(7,[study.f32(x) for x in delta*2] if update=="train" else [0.]*(2*len(metrics)))
                report["cases"].append(row)
        qualification = copy.deepcopy(reference["calibration_verification"]["cases"][0])
        qualification.update(name=request["cases"][-2]["name"],mixture=copy.deepcopy(request["cases"][-2]["mixture"]))
        del qualification["pair_metric"]
        reference["calibration_verification"]["cases"].append(qualification)
    raw = study.encoded(request)
    for report in (reference,actual):
        report.update(schema="spiraltorch.byte_corpus.result.v7",request_sha256=study.digest(raw))
    return raw,reference,actual,source_raw,retained


def packet_fixture():
    raw,reference,_,source,_ = fixture()
    request = study.decode_json(raw)
    template_report = study.decode_json(headwise_packet()[1]["report_json"])
    template = study.decode_json(template_report["cases"][0]["model_json"])
    records = []
    for case,measured in zip(request["cases"][9::2],reference["calibration_verification"]["cases"][2:]):
        model = copy.deepcopy(template)
        model["schema"] = "spiraltorch.nn.byte_decoder_checkpoint.v4"
        geometry = model["model"]["geometry"]
        del geometry["head_pair_metrics"]
        geometry["mixture"] = dict(metrics=case["mixture"]["metrics"],scales=case["mixture"]["scales"],raw_logits=[case["parameters"][7]["values"]])
        for stored,initial in zip(resume.stored_parameters(model,0,mixture=case["mixture"]),case["parameters"]):
            stored["values"][:] = initial["values"]
        checkpoint = study.encoded(model).decode()
        records.append(dict(name=case["name"],seed=case["seed"],mixture=case["mixture"],no_updates_consumed=True,
            before_checkpoint_json=checkpoint,fitted_checkpoint_json=checkpoint,raw_gains=[case["parameters"][6]["values"]],
            reference_rms=measured["reference_rms"],candidate_rms=measured["reference_rms"],fitted_rms=measured["fitted_rms"],
            realized_relative_errors=measured["realized_relative_errors"],valid_pairs_per_head=measured["valid_pairs_per_head"]))
    report = dict(schema="spiraltorch.byte_corpus.mixture_preparation.v1",source_request_sha256=study.digest(source),
        request_sha256=study.digest(raw),adapter="fabricated test data",no_updates_consumed=True,
        train_batch_indices=request["bias_calibration"]["train_batch_indices"],relative_tolerance=1e-5,cases=records)
    return source,dict(schema="spiraltorch.byte_corpus.prepared.v1",request_json=raw.decode(),report_json=study.encoded(report).decode())


class MixtureStudy(unittest.TestCase):
    def test_seventeen_modes_and_four_contrasts(self):
        raw,reference,actual,_,_ = fixture()
        request = study.decode_json(raw)
        self.assertEqual(study.request_version(request),7)
        for _ in range(17):
            request["cases"] = request["cases"][1:]+request["cases"][:1]
            self.assertEqual(study.request_version(request),7)
        result = study.compare(raw,reference,actual)
        self.assertEqual(len(result["mixture_contrasts"]),4)
        self.assertTrue(all(all(c["same_sign_as_torch"].values()) for c in result["mixture_contrasts"]))

    def test_reject_missing_foreign_or_ambiguous_identity(self):
        for mutation in ("null","duplicate","both","headwise","missing","legacy","source","metadata","seed","geometry"):
            request = study.decode_json(fixture()[0])
            case = request["cases"][9]
            if mutation=="null": case["mixture"] = None
            if mutation=="duplicate": request["cases"][11]["mixture"] = copy.deepcopy(case["mixture"])
            if mutation=="both": case["pair_metric"] = mixture.P
            if mutation=="headwise": case["head_pair_metrics"] = [[mixture.P,mixture.E]]
            if mutation=="missing": del case["mixture"]
            if mutation=="legacy": request["schema"] = study.REQUEST_V5
            if mutation=="source": request["mixture_controls"]["source_request_sha256"] = "?"
            if mutation=="metadata": del request["mixture_controls"]
            if mutation=="seed": case["seed"] = 999
            if mutation=="geometry": case["geometry_update"] = "frozen"
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                study.request_version(request)

    def test_units_uniform_router_and_every_initial_twin_bit(self):
        for mutation in ("unit","negative","zero","nan","router","signed_zero","backbone","gain","shape","coercion"):
            request = study.decode_json(fixture()[0])
            case = request["cases"][9]
            if mutation=="unit": case["mixture"]["scales"][0][0] *= 1.01
            if mutation=="negative": case["mixture"]["scales"][0][0] = -1.
            if mutation=="zero": case["mixture"]["scales"][0][0] = 0.
            if mutation=="nan": case["mixture"]["scales"][0][0] = float("nan")
            if mutation=="router": case["parameters"][7]["values"][0] = .01
            if mutation=="signed_zero": case["parameters"][7]["values"][0] = -0.
            if mutation=="backbone": case["parameters"][0]["values"][0] += .01
            if mutation=="gain": case["parameters"][6]["values"][0] += .01
            if mutation=="shape": case["parameters"][7]["shape"] = [2.,2]
            if mutation=="coercion": case["parameters"][7]["values"][0] = True
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                study.request_version(request)

    def test_individual_router_component_sensitivity_and_freeze(self):
        self.assertEqual(len(mixture.router_deltas([0.]*4,[10.,1e-5,-10.,-1e-5],[10.,1e-5,-10.,-1e-5],[2,2])),4)
        for actual,expected in (([10.,0.],[10.,1e-5]),([10.,1e-9],[10.,1e-9])):
            with self.assertRaises(ValueError):
                mixture.router_deltas([0.,0.],actual,expected,[1,2])
        for mutation in ("missing_delta","frozen","policy"):
            raw,ref,actual,_,_ = fixture()
            if mutation=="missing_delta": actual["cases"][9]["final_parameters"][7][1] = 0.
            if mutation=="frozen": actual["cases"][10]["final_parameters"][7][0] = 1e-7
            if mutation=="policy": actual["cases"][10]["mixture"]["router_update"] = "train"
            with self.assertRaises(ValueError):
                study.compare(raw,ref,actual)

    def test_qualification_covers_every_subset_and_shared_target(self):
        raw,ref,_,_,_ = fixture()
        request = study.decode_json(raw)
        calibration.verify_qualification(request,ref["calibration_verification"])
        for mutation in ("missing","units","target","count","gate"):
            record = copy.deepcopy(ref["calibration_verification"])
            row = record["cases"][2]
            if mutation=="missing": record["cases"].pop()
            if mutation=="units": row["mixture"]["scales"][0][0] += .1
            if mutation=="target":
                row["reference_rms"][0][0] *= 2
                row["fitted_rms"][0][0] *= 2
            if mutation=="count": row["valid_pairs_per_head"] += 1
            if mutation=="gate": row["fitted_rms"][0][1] *= 1.01
            with self.assertRaises(ValueError):
                calibration.verify_qualification(request,record)

    def test_fixed_router_keeps_full_non_router_pullbacks(self):
        raw,_,full,source,retained = fixture()
        first = copy.deepcopy(full)
        first["schema"] = "spiraltorch.byte_corpus.partial.v7"
        for row in first["cases"]:
            row["training"] = row["training"][:1]
            row["validation"] = row["validation"][:1]
        checked = controls.verify(raw,source,full,retained,first)
        self.assertEqual(len(checked["inherited_full_reports_exact"]),9)
        self.assertEqual(len(checked["pairs"]),8)
        for mutation in ("backbone","encoder","gain","frozen","history"):
            bad = copy.deepcopy(first)
            row = bad["cases"][10]
            if mutation=="history": row["training"] = []
            else: row["final_parameters"][dict(backbone=0,encoder=2,gain=6,frozen=7)[mutation]][0] += .01
            with self.assertRaises(ValueError):
                controls.verify(raw,source,full,retained,bad)

    def test_preparation_binds_initial_bits_gain_arithmetic_and_topology(self):
        source,packet = packet_fixture()
        self.assertTrue(preparation.verify(source,packet)["passed"])
        for mutation in ("missing","units","gain","ratio","rms","count","source","router","curvature","heads","stage"):
            bad = copy.deepcopy(packet)
            report = study.decode_json(bad["report_json"])
            row = report["cases"][0]
            if mutation=="missing": report["cases"].pop()
            if mutation=="units": row["mixture"]["scales"][0][0] *= 1.01
            if mutation=="gain": row["raw_gains"][0][0] += .01
            if mutation=="ratio": row["candidate_rms"][0][0] *= 2
            if mutation=="rms": row["fitted_rms"][0][1] *= 1.01
            if mutation=="count": row["valid_pairs_per_head"] += 1
            if mutation=="source": report["source_request_sha256"] = "b"*64
            if mutation in ("router","curvature","heads","stage"):
                model = study.decode_json(row["fitted_checkpoint_json"])
                if mutation=="router": model["model"]["geometry"]["mixture"]["raw_logits"][0][0] = .01
                if mutation=="curvature": model["model"]["geometry"]["curvature"] = -.5
                if mutation=="heads": model["model"]["blocks"][0]["heads"] = 1
                if mutation=="stage": model["model"]["geometry"]["projection"]["stages"] = []
                row["fitted_checkpoint_json"] = study.encoded(model).decode()
            bad["report_json"] = study.encoded(report).decode()
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                preparation.verify(source,bad)

    def test_checkpoint_extractor_binds_mixture_version_units_and_shape(self):
        _,packet = packet_fixture()
        case = study.decode_json(packet["request_json"])["cases"][9]
        row = study.decode_json(packet["report_json"])["cases"][0]
        model = study.decode_json(row["fitted_checkpoint_json"])
        self.assertEqual(len(resume.stored_parameters(model,0,mixture=case["mixture"])),len(case["parameters"]))
        for mutation in ("version","units","route","blocks","shape","revision"):
            bad = copy.deepcopy(model)
            mix = bad["model"]["geometry"]["mixture"]
            if mutation=="version": bad["schema"] = "spiraltorch.nn.byte_decoder_checkpoint.v3"
            if mutation=="units": mix["scales"][0][0] *= 2
            if mutation=="route": mix["metrics"].reverse()
            if mutation=="blocks": mix["raw_logits"].pop()
            if mutation=="shape": mix["raw_logits"][0].pop()
            if mutation=="revision": bad["attempted_revision"] = "1"
            with self.assertRaises(ValueError):
                resume.stored_parameters(bad,0,mixture=case["mixture"])


if __name__=="__main__":
    unittest.main()
