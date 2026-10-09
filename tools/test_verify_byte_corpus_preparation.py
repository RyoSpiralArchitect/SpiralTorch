import copy
import unittest

import byte_corpus_study as study
import verify_byte_corpus_preparation as verify
from test_byte_corpus_calibration import fixture as calibrated_fixture
from test_byte_corpus_study import metric_fixture
from test_verify_byte_corpus_resume import model_fixture


def fixture():
    source_raw = metric_fixture()[0]
    raw, _, _ = calibrated_fixture()
    request = study.decode_json(raw)
    # Rust's preparer emits no trailing newline inside the opaque request.
    raw = raw.rstrip(b"\n")
    checkpoints = []
    for case in [request["cases"][3], request["cases"][5]]:
        model, _ = model_fixture(True)
        model["attempted_revision"] = "0"
        model["schema"] = "spiraltorch.nn.byte_decoder_checkpoint.v2"
        model["model"]["geometry"]["pair_metric"] = study.FLAT
        parameters = verify.stored_parameters(model, 0, study.FLAT)
        for actual, expected in zip(parameters, case["parameters"]):
            actual["values"][:] = expected["values"]
        checkpoints.append(study.encoded(model).decode())
    report = dict(schema="spiraltorch.byte_corpus.bias_preparation.v1", adapter="synthetic, not evidence",
                  bias_calibration=request["bias_calibration"], request_sha256=study.digest(raw), cases=[dict(
                      seed=11, valid_pairs_per_head=6, raw_gains=[[.25]], reference_rms=[[2.]],
                      candidate_rms=[[1.]], fitted_rms=[[2.]], realized_relative_errors=[[0.]],
                      before_checkpoint_json=checkpoints[0], fitted_checkpoint_json=checkpoints[1], no_updates_consumed=True)])
    packet = dict(schema="spiraltorch.byte_corpus.prepared.v1", request_json=raw.decode(), report_json=study.encoded(report).decode())
    return source_raw, packet


class Preparation(unittest.TestCase):
    def test_checkpoint_types_are_not_coerced_to_float32(self):
        for parameter in (dict(shape=[1], values=[True]), dict(shape=[1.0], values=[1.0]),
                          dict(shape=[True], values=[1.0])):
            with self.assertRaises(ValueError):
                verify.same_parameters([parameter], [dict(shape=[1], values=[1.0])])

    def test_source_and_actual_initial_checkpoint_binding(self):
        source, packet = fixture()
        self.assertTrue(verify.verify(source, packet)["passed"])
        with self.assertRaises(ValueError):
            verify.verify(source + b"\n", packet)

    def test_request_mutations_reject_even_when_packet_hash_is_rebound(self):
        for mutation in ("source", "case", "data", "nongain", "gain", "gain_pair"):
            source, packet = fixture()
            request = study.decode_json(packet["request_json"])
            if mutation == "source":
                request["bias_calibration"]["source_request_sha256"] = "b" * 64
            elif mutation == "case":
                request["cases"][0]["name"] = "relabeled"
            elif mutation == "data":
                request["rate"] *= 2
            elif mutation == "nongain":
                request["cases"][5]["parameters"][2]["values"][0] += .01
            elif mutation == "gain":
                request["cases"][5]["parameters"][6]["values"][0] += .01
            else:
                for c in request["cases"][5:]:
                    c["parameters"][6]["values"][0] += .01
            packet["request_json"] = study.encoded(request).decode()
            report = study.decode_json(packet["report_json"])
            report["request_sha256"] = study.digest(packet["request_json"].encode())
            report["bias_calibration"] = request["bias_calibration"]
            packet["report_json"] = study.encoded(report).decode()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                verify.verify(source, packet)

    def test_report_coverage_arithmetic_and_checkpoint_mutations_reject(self):
        for mutation in ("seed", "count", "zero", "gate", "claim", "head", "gains", "initial", "revision", "updates"):
            source, packet = fixture()
            report = study.decode_json(packet["report_json"])
            record = report["cases"][0]
            if mutation == "seed":
                record["seed"] = 12
            elif mutation == "count":
                record["valid_pairs_per_head"] = 7
            elif mutation == "zero":
                record["candidate_rms"] = [[0.]]
            elif mutation == "gate":
                record["fitted_rms"] = [[2.1]]
            elif mutation == "claim":
                record["realized_relative_errors"] = [[1e-7]]
            elif mutation == "head":
                record["reference_rms"] = [[]]
            elif mutation == "gains":
                record["raw_gains"][0][0] += 1e-7
            elif mutation in ("initial", "revision"):
                cp = study.decode_json(record["fitted_checkpoint_json"])
                if mutation == "initial":
                    cp["model"]["geometry"]["raw_gains"][0][0] += 1e-7
                else:
                    cp["attempted_revision"] = "1"
                record["fitted_checkpoint_json"] = study.encoded(cp).decode()
            else:
                record["no_updates_consumed"] = False
            packet["report_json"] = study.encoded(report).decode()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                verify.verify(source, packet)


if __name__ == "__main__":
    unittest.main()
