#!/usr/bin/env python3
"""Admit an explicit terminal-forward comparison, never as ordinary forwarding."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_graph_forward_paths import admit_native, digest
from validate_module_forward_paths import validate_python


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fixture', 'baseline-python', 'candidate-python', 'baseline-receipt', 'candidate-receipt', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    paths = [args.fixture, args.baseline_python, args.candidate_python, args.baseline_receipt, args.candidate_receipt,
             Path(__file__), Path(__file__).with_name('bench_graph_forward_paths.py'),
             Path(__file__).with_name('validate_module_forward_paths.py')]
    report = dict(schema='spiraltorch.module_terminal_paths_validation.v1', status='error',
        module_apis=dict(baseline='forward_then_snapshot',candidate='forward_snapshot'),
        boundary='Separate serial Python processes with eager Torch CPU/MPS controls. The candidate uses explicit terminal capture; ordinary GPU-output forwarding remains separate. Completed-read and eight-forward/final-read-only burst are distinct workloads. Includes host wrapper costs; no fastest-Torch or isolated GPU queue-cost claim.')
    with args.output.open('x') as out:
        try:
            sources = {str(p.resolve()):digest(p) for p in paths}
            native, old, new, baseline, candidate = [json.loads(p.read_bytes()) for p in paths[:5]]
            admit_native(native)
            for receipt, root, data in [(baseline,args.baseline_receipt.parent,old),(candidate,args.candidate_receipt.parent,new)]:
                if receipt.get('status') != 'passed': raise ValueError('runtime not verified')
                for name, sha in receipt['products'].items():
                    if digest(root/name) != sha: raise ValueError('frozen product changed: '+name)
                library = root/'python-release.dylib'
                if data['sources'].get(str(library.resolve())) != receipt['products']['python-release.dylib']:
                    raise ValueError('native library identity differs')
            a = validate_python(old,native,args.fixture)
            b = validate_python(new,native,args.fixture,terminal_capture=True)
            report['python'] = [dict(shape=n['shape'],depth=n['depth'],seed=n['seed'],baseline=o,candidate=n,
                candidate_over_baseline={route:n['summary'][route]['median_ms_per_forward']/o['summary'][route]['median_ms_per_forward']
                    for route in ('module_resident_d2h','module_resident_burst','python_scalar_burst','torch_mps_burst')})
                for o,n in zip(a,b,strict=True)]
            if sources != {str(p.resolve()):digest(p) for p in paths}: raise ValueError('sources changed')
            report.update(status='passed',sources=sources,baseline_source=baseline['source'],candidate_source=candidate['source'])
        except BaseException as error:
            report['error'] = repr(error)
            raise
        finally:
            json.dump(report,out,indent=2,allow_nan=False); out.write('\n')


if __name__ == '__main__':
    main()
