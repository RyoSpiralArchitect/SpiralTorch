"""Synthetic API-boundary admission checks, not GPU or timing evidence."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_graph_forward_paths import digest, requests, summarize
from validate_module_forward_paths import validate_python


class Boundaries(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = (Path(self.directory.name)/'fixture.json').resolve()
        self.native = dict(cases=[dict(seed=seed,shape=shape,depth=depth,reference=[.25]) for seed,shape,depth in requests()])
        self.path.write_text(json.dumps(self.native))
        routes = [f'python_{k}_{c}' for k in ('scalar','register') for c in ('h2h','burst')]
        routes += [f'torch_{d}_{c}' for d in ('cpu','mps') for c in ('h2h','burst')]
        routes += ['module_resident_d2h','module_resident_burst']
        self.doc = dict(schema='spiraltorch.module_forward_paths.v1',status='passed',sources={str(self.path):digest(self.path)},
                        devices=['cpu','mps'],build_info=dict(profile='release',features=dict(wgpu=True)),cases=[])
        for frozen in self.native['cases']:
            samples=[]
            for block in range(9):
                offset=(block+3+frozen['seed'])%len(routes)
                order=routes[offset:]+routes[:offset]
                samples.extend(dict(block=block,route=r,order=order,forwards=8 if r.endswith('_burst') else 1,
                                    elapsed_ms=1.,max_abs_error=0.) for r in order)
            self.doc['cases'].append(dict(shape=frozen['shape'],depth=frozen['depth'],seed=frozen['seed'],routes=routes,
                module_cache=dict(compilations=1,cache_hits=108,submitted_forwards=109),module_cold_ms=1.,samples=samples,
                last_outputs={r:[.25] for r in routes},summary=summarize(samples)))

    def test_distinct_modes_require_their_own_schema_and_api(self):
        self.assertEqual(len(validate_python(self.doc,self.native,self.path)),9)
        with self.assertRaises(ValueError): validate_python(self.doc,self.native,self.path,terminal_capture=True)
        self.doc.update(schema='spiraltorch.module_terminal_forward_paths.v1',module_api='forward_snapshot')
        self.assertEqual(len(validate_python(self.doc,self.native,self.path,terminal_capture=True)),9)
        with self.assertRaises(ValueError): validate_python(self.doc,self.native,self.path)
        self.doc['module_api']='forward_then_snapshot'
        with self.assertRaises(ValueError): validate_python(self.doc,self.native,self.path,terminal_capture=True)

    def test_wrong_api_cannot_hide_inside_the_ordinary_schema(self):
        self.doc['module_api']='forward_snapshot'
        with self.assertRaises(ValueError): validate_python(self.doc,self.native,self.path)

    def test_terminal_mode_still_checks_outputs_counters_and_all_samples(self):
        self.doc.update(schema='spiraltorch.module_terminal_forward_paths.v1',module_api='forward_snapshot')
        for mutate in (lambda d:d['cases'][0]['samples'].pop(),
                       lambda d:d['cases'][0]['module_cache'].__setitem__('submitted_forwards',1),
                       lambda d:d['cases'][0]['last_outputs']['module_resident_d2h'].__setitem__(0,2.)):
            bad=copy.deepcopy(self.doc);mutate(bad)
            with self.assertRaises(ValueError): validate_python(bad,self.native,self.path,terminal_capture=True)


if __name__ == '__main__':
    unittest.main()
