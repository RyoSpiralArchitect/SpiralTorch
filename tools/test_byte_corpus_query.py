"""Fabricated contextual protocol controls, not corpus-learning evidence."""
import copy
import unittest

import byte_corpus_study as study
import byte_corpus_query as query
from test_byte_corpus_mixture import fixture as shared_fixture


def fixture():
    source_raw, reference, actual, _, _ = shared_fixture()
    request = study.decode_json(source_raw)
    sources = [copy.deepcopy(c) for c in request["cases"]
               if c.get("mixture", {}).get("router_update") == "train"]
    for source in sources:
        for update in ("train", "frozen"):
            c = copy.deepcopy(source)
            c["name"] += "-query-" + update
            c["mixture"]["query_projection"] = dict(weight_update=update, bias="frozen_zero")
            output = request["config"]["heads"] * len(c["mixture"]["metrics"])
            cols = request["config"]["geometry_cols"]
            parameters = [
                dict(name="geometry.query_projection.0.weight", shape=[cols,output], values=[0.]*(cols*output)),
                dict(name="geometry.query_projection.0.bias", shape=[output], values=[0.]*output)]
            c["parameters"][8:8] = parameters
            request["cases"].append(c)
            for report in (reference, actual):
                row = copy.deepcopy(next(r for r in report["cases"] if r["name"] == source["name"]))
                row.update(name=c["name"], mixture=copy.deepcopy(c["mixture"]),
                           parameter_tensors=len(c["parameters"]),
                           parameter_scalars=sum(len(p["values"]) for p in c["parameters"]),
                           trainable_parameter_scalars=study.trainable_scalars(c))
                row["final_parameters"][8:8] = [
                    [study.f32(.001 if update == "train" else 0.)]*(cols*output), [0.]*output]
                report["cases"].append(row)
            qualification = copy.deepcopy(next(r for r in reference["calibration_verification"]["cases"]
                                              if r.get("name") == source["name"]))
            qualification.update(name=c["name"], mixture=copy.deepcopy(c["mixture"]))
            reference["calibration_verification"]["cases"].append(qualification)
    request.update(schema=study.REQUEST_V8, query_controls={"source_request_sha256":study.digest(source_raw)})
    raw = study.encoded(request)
    for report in (reference,actual):
        report.update(schema="spiraltorch.byte_corpus.result.v8", request_sha256=study.digest(raw))
    return source_raw,raw,reference,actual


class ContextualStudy(unittest.TestCase):
    def test_complete_controls_and_frozen_bias(self):
        _,raw,ref,actual = fixture()
        request = study.decode_json(raw)
        self.assertEqual(study.request_version(request),8)
        for _ in range(25):
            request["cases"] = request["cases"][1:]+request["cases"][:1]
            self.assertEqual(study.request_version(request),8)
        result = study.compare(raw,ref,actual)
        self.assertEqual(len(result["query_contrasts"]),4)
        self.assertEqual(len(result["mixture_contrasts"]),4)
        for index in (17,18):
            c = study.decode_json(raw)["cases"][index]
            self.assertTrue(study.parameter_frozen(c,"geometry.query_projection.0.bias"))
            self.assertEqual(study.parameter_frozen(c,"geometry.query_projection.0.weight"),index==18)
            self.assertFalse(study.parameter_frozen(c,"geometry.mixture_logits.0"))

    def test_policy_and_initialization_rejections(self):
        for mutation in ("null","bias","weight","duplicate","missing","order","units","backbone",
                         "zero","signed_zero","legacy","hash"):
            request = study.decode_json(fixture()[1])
            c = request["cases"][17]
            if mutation=="null": c["mixture"]["query_projection"] = None
            if mutation=="bias": c["mixture"]["query_projection"]["bias"] = "train"
            if mutation=="weight": c["mixture"]["query_projection"]["weight_update"] = "detach"
            if mutation=="duplicate": request["cases"][18]["mixture"] = copy.deepcopy(c["mixture"])
            if mutation=="missing": request["cases"].pop()
            if mutation=="order": c["parameters"][8:10] = c["parameters"][9:7:-1]
            if mutation=="units": c["mixture"]["scales"][0][0] *= 2
            if mutation=="backbone": c["parameters"][0]["values"][0] += .01
            if mutation=="zero": c["parameters"][8]["values"][0] = .01
            if mutation=="signed_zero": c["parameters"][9]["values"][0] = -0.
            if mutation=="legacy": request["schema"] = study.REQUEST_V7
            if mutation=="hash": request["query_controls"]["source_request_sha256"] = "bad"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                study.request_version(request)

    def test_dropped_training_and_frozen_mutations_fail(self):
        for index,slot in ((17,8),(17,9),(18,8),(18,9)):
            _,raw,ref,actual = fixture()
            values = actual["cases"][index]["final_parameters"][slot]
            values[:] = [0. if index==17 and slot==8 else .000001]*len(values)
            with self.subTest(index=index,slot=slot), self.assertRaises(ValueError):
                study.compare(raw,ref,actual)

    def test_preparation_preserves_exact_source(self):
        source,raw,_,_ = fixture()
        report = dict(schema="spiraltorch.byte_corpus.query_preparation.v1",
                      source_request_sha256=study.digest(source),request_sha256=study.digest(raw),
                      no_fitting=True,no_updates_consumed=True)
        packet = dict(schema="spiraltorch.byte_corpus.prepared.v1",
                      request_json=raw.decode(),report_json=study.encoded(report).decode())
        self.assertEqual(query.verify_preparation(source,packet)[0],raw)
        request = study.decode_json(raw)
        request["rate"] *= .5
        raw = study.encoded(request)
        packet["request_json"] = raw.decode()
        report["request_sha256"] = study.digest(raw)
        packet["report_json"] = study.encoded(report).decode()
        with self.assertRaises(ValueError):
            query.verify_preparation(source,packet)


if __name__ == "__main__":
    unittest.main()
