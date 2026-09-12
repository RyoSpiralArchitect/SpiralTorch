"""CPU-only admission tests; fabricated timings are never benchmark evidence."""
import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_graph_forward_paths import requests
from validate_module_direct_io import ROUTES, validate_browser


def fixture():
    native = {"cases": []}
    browser = dict(schema="spiraltorch.module_direct_io_matched.v1",status="passed",
        page_errors=[],fixture_request="nn-module-matched",routes=ROUTES,
        warmup=3,samples_per_route=9,burst=8,cases=[])
    for seed,shape,depth in requests():
        base = dict(seed=seed,shape=shape,depth=depth)
        native["cases"].append(dict(base,reference=[.25]))
        samples = []
        for block in range(9):
            offset=(block+3+seed)%len(ROUTES)
            order=ROUTES[offset:]+ROUTES[:offset]
            samples.extend(dict(block=block,route=r,order=order,
                forwards=8 if r.endswith("_burst") else 1,elapsed_ms=1.,max_abs_error=0.) for r in order)
        browser["cases"].append(dict(base,routes=ROUTES,samples=samples,
            last_outputs={r:[.25] for r in ROUTES},cold_ms={v:1. for v in ("baseline","candidate")},
            adapters={v:dict(backend="BrowserWebGpu",device_type="Other") for v in ("baseline","candidate")},
            cache={v:dict(compilations="1",cache_hits="108",submitted_forwards="109") for v in ("baseline","candidate")}))
    return native,browser


class Admission(unittest.TestCase):
    def test_complete_matrix(self):
        native,browser=fixture()
        rows=validate_browser(browser,native)
        self.assertEqual(len(rows),9)
        self.assertTrue(all(row["candidate_over_baseline"]["module_burst"]==1. for row in rows))

    def test_incomplete_or_cpu_matrix(self):
        native,browser=fixture()
        variants=[]
        for key,value in [("warmup",0),("routes",ROUTES[:-1]),("page_errors",["lost"]),("cases",browser["cases"][:-1])]:
            changed=copy.deepcopy(browser);changed[key]=value;variants.append(changed)
        changed=copy.deepcopy(browser);changed["cases"][0]["adapters"]["candidate"]["device_type"]="Cpu";variants.append(changed)
        for changed in variants:
            with self.assertRaises(ValueError):validate_browser(changed,native)

    def test_bad_outputs_counters_and_timing(self):
        native,browser=fixture()
        for kind in ("values","counter","nan","zero","order","burst"):
            changed=copy.deepcopy(browser);case=changed["cases"][0]
            if kind=="values":case["last_outputs"][ROUTES[0]]=[1.]
            if kind=="counter":case["cache"]["baseline"]["cache_hits"]="0"
            if kind=="nan":case["samples"][0]["elapsed_ms"]=float("nan")
            if kind=="zero":case["samples"][0]["elapsed_ms"]=0.
            if kind=="order":case["samples"][0]["order"]=[]
            if kind=="burst":case["samples"][0]["forwards"]+=1
            with self.subTest(kind=kind),self.assertRaises(ValueError):validate_browser(changed,native)


if __name__ == "__main__":unittest.main()
