"""Standard-library tests, plus an optional finite-difference check on the artificial archive."""
import copy
import json
from pathlib import Path
import runpy
import struct
import sys
import unittest

S = runpy.run_path(str(Path(__file__).resolve().with_name("sweep.py")))


def pack(*xs):
    return S["b64"](struct.pack("<%df" % len(xs), *xs))


def observation(g, b, y):
    return dict(gate=pack(*g), bias=pack(*b), y=pack(*y))


class TestRules(unittest.TestCase):
    def test_norm_angle_and_forward_denominator(self):
        baseline = observation([1,0], [0,1], [2,2])
        trial = observation([2,0], [0,2], [3,2])
        row = {"receipt":{"site_input":pack(1,2), "arms":{"affine":{"gate":pack(4,0),"bias":pack(0,4)}}}}
        m = S["compare"](row, baseline, trial, 2)
        self.assertEqual(m["min_vs_affine"], 0.5)
        self.assertEqual(m["min_vs_one"], 2.0)
        self.assertAlmostEqual(m["min_cosine"], 1.0)
        self.assertTrue(m["passed"])
        self.assertEqual(m["forward_shift"], 1.0)
        self.assertLess(S["direction"]([1],[-1]), 0)
        self.assertIsNone(S["ratio"](1,0))
        self.assertIsNone(S["direction"]([1],[0]))

    def test_same_row_both_hosts_and_smallest_radius(self):
        samples = [dict(min_vs_affine=.9, min_vs_one=1, min_cosine=.999),
                   dict(min_vs_affine=.1, min_vs_one=3, min_cosine=.999),
                   dict(min_vs_affine=.9, min_vs_one=3, min_cosine=.98),
                   dict(min_vs_affine=.9, min_vs_one=3, min_cosine=None)]
        self.assertFalse(any(map(S["passes"], samples)))
        def row(ok):
            m = dict(samples[0], min_vs_one=4 if ok else None)
            return dict(m, passed=S["passes"](m), blocks={b:m for b in S["BLOCKS"]},
                        forward_shift=0, forward_shift_vs_input=0)
        groups = {(r,h):[row(True)]*2+[row(False)]*2 for r in S["RADII"] for h in S["HOSTS"]}
        groups[2,"affine"] = [row(True)]*4
        self.assertIsNone(S["aggregate"](groups,3)[1])
        for r in (0.5,4,8):
            for h in S["HOSTS"]:
                groups[r,h] = [row(True)]*3+[row(False)]
        t, selected = S["aggregate"](groups,3)
        self.assertEqual(selected, 4)
        self.assertEqual(t["4"]["both"]["min_vs_one"]["defined"], 3)

    def test_capture_width_nonfinite_and_sign(self):
        for raw in (b"abc", struct.pack("<f", float("nan"))):
            with self.assertRaises(ValueError):
                S["vector"](raw,1)
        row = dict(key="k",host="both",receipt={"arms":{"both":{"gate":pack(+0.),"bias":pack(.5)}}})
        with self.assertRaisesRegex(ValueError, "R=1 mismatch"):
            S["assert_anchor"]({"gate":pack(-0.),"bias":pack(.5)}, row)

    def test_only_fixed_scalar_may_change(self):
        rec = {"placements":[{"recipe":{"strength":.25}, "tensors":[{
            "name":"log_radius","shape":[],"values":[0.],"requires_grad":False,"role":"buffer"}]}]}
        saved = copy.deepcopy(rec)
        trial = S["radius_copy"](rec, 4)
        S["check_variant"](rec, trial, 4)
        self.assertEqual(saved,rec)
        trial["placements"][0]["recipe"]["strength"] = .5
        with self.assertRaises(ValueError):
            S["check_variant"](rec, trial, 4)
        S["log_buffer"](saved)["requires_grad"] = True
        with self.assertRaises(ValueError):
            S["radius_copy"](saved, 4)


def finite_difference(archive):
    from spiraltorch import GeometryAdapterArtifact as Artifact, NativeGeometryRuntime as Runtime, _rs
    p, rows, _ = S["inputs"](archive, "rehearsal")
    S["need"](S["sha"](Path(_rs.__file__).read_bytes()) == p["extension"], "wrong test extension")
    rec = S["check_parent"](p)
    row = rows[0]["receipt"]
    width = p["shape"][1]
    x, dy = (S["vector"](S["unpack"](row[k]),width) for k in ("site_input", "upstream"))
    def forward(record):
        return Runtime(Artifact.create_json(json.dumps(record))).forward_buffer(p["site"],x,1)
    errs = []
    for r in (.5, 4):
        state = S["radius_copy"](rec,r)
        vjp = forward(state).vjp_parameters_buffer(dy)
        for b in S["BLOCKS"]:
            g = S["vector"](vjp[b], width)
            i = max(range(width), key=lambda j:abs(g[j]))
            dots, coords = [], []
            for sign in (-1,1):
                perturbed = copy.deepcopy(state)
                vals = S["tensor_at"](perturbed,b)["values"]
                vals[i] = S["f32"](vals[i] + sign/512.)
                coords.append(vals[i])
                dots.append(sum(a*z for a,z in zip(forward(perturbed).output,dy)))
            difference = (dots[1]-dots[0])/(coords[1]-coords[0])
            S["need"](abs(g[i]) > 1e-8, "degenerate synthetic derivative")
            rel = abs(difference-g[i])/abs(g[i])
            S["need"](rel <= .01, "synthetic finite difference failed: " + str((r,b,i,rel)))
            errs.append((r,b,i,rel))
    print(json.dumps(errs))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--toy-finite-difference":
        finite_difference(Path(sys.argv[2]))
    else:
        unittest.main()
