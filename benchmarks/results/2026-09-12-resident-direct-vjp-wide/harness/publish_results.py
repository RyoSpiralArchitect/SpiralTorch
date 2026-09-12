"""Publish all wide-stratum results, retaining tensor payloads only locally."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
log = Path(__file__).resolve().parent
run = log / 'measurement-a'
out = args.output.resolve()
receipt = read(run / 'receipt.json')
assert receipt['status'] == 'passed' and 'active' not in receipt
assert len(receipt['steps']) == 18 and all(s['exit_code'] == 0 for s in receipt['steps'])
assert sha(log / 'measure.py') == receipt['harness_sha256']
for path, expected in receipt['products'].items():
    assert sha(Path(path)) == expected, path
for source in [receipt['harness_source'], *receipt['workers'].values()]:
    tree = subprocess.check_output(['git', 'rev-parse', source['commit'] + '^{tree}'],
        cwd=args.repo, text=True).strip()
    assert tree == source['tree']
out.mkdir()
origins = {}


def copy(path, name):
    target = out / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
    assert sha(path) == sha(target)
    origins[name] = str(path)


inventory = []
for path in sorted(run.rglob('*')):
    if path.is_file():
        inventory.append(dict(path=str(path.relative_to(run)), bytes=path.stat().st_size,
            sha256=sha(path)))
write(out / 'raw-inventory.json', dict(schema='spiraltorch.local_raw_inventory.v1',
    root=str(run), records=inventory, bytes=sum(r['bytes'] for r in inventory),
    published_raw_tensor_payloads=False,
    boundary='Full terminal measurement originals remain local; hashes are not raw payload publication.'))
inventory_hashes = {str(run / r['path']): r['sha256'] for r in inventory}
rows, timings = [], []
for repetition in ['a', 'b']:
    for mode in ['plain', 'topos_ema', 'clipped_topos_ema']:
        name = repetition + '-' + mode + '-validation.json'
        data = read(run / name)
        assert data['status'] == 'passed' and data['matrix'] == 'wide'
        assert len(data['cases']) == 9 and data['browser_intervals_revalidated'] == 360
        assert data['learner_optimizer'] == (None if mode == 'plain' else mode)
        for item in data['inputs']:
            assert inventory_hashes.get(item['path']) == item['sha256'], item['path']
        for role, source in receipt['workers'].items():
            binding = data['source_bindings'][role]
            assert binding['valid'] and all(binding['checks'].values())
            assert binding['source_commit'] == source['commit']
            assert binding['source_tree'] == source['tree']
        copy(run / name, 'run/' + name)
        for case in data['cases']:
            timings.append(dict(run=repetition, mode=mode, config=case['config'],
                native=case['native'], browser=case['browser'],
                max_abs_error=max(case['max_abs_errors'].values())))
        for cadence in ['immediate', 'deferred']:
            row = dict(run=repetition, mode=mode, cadence=cadence)
            for route in ['native', 'browser']:
                cases = [case[route][cadence] for case in data['cases']]
                ratios = [case['baseline_over_candidate'] for case in cases]
                assert all(math.isfinite(ratio) and ratio > 0 for ratio in ratios)
                row[route] = dict(min=min(ratios), max=max(ratios),
                    median_case_ratio=statistics.median(ratios),
                    geomean=math.exp(sum(map(math.log, ratios)) / len(ratios)),
                    candidate_faster=sum(ratio > 1 for ratio in ratios))
                if route == 'native':
                    torch = [case['torch_over_candidate'] for case in cases]
                    row[route].update(torch_faster=sum(ratio < 1 for ratio in torch),
                        torch_over_candidate_range=[min(torch), max(torch)])
            rows.append(row)
copy(run / 'receipt.json', 'run/receipt.json')
for path in sorted(run.glob('*.log')):
    assert path.stat().st_size < 5 * 1024 * 1024
    copy(path, 'run/' + path.name)
for path in sorted(run.glob('*.stderr')):
    assert path.stat().st_size < 5 * 1024 * 1024
    copy(path, 'run/' + path.name)
for name in ['measure.py', 'measurement-a.log', 'review_checks.py', 'publish_results.py']:
    copy(log / name, 'harness/' + name)
copy(log / 'verify_results.py', 'verify.py')
copy(log / 'evidence-wide-README.md', 'README.md')
for name, expected in receipt['unchanged_helpers'].items():
    versions = [subprocess.check_output(['git', 'show', source['commit'] + ':' + name], cwd=args.repo)
        for source in [receipt['harness_source'], *receipt['workers'].values()]]
    assert versions[0] == versions[1] == versions[2]
    assert hashlib.sha256(versions[0]).hexdigest() == expected
    target = out / 'sources' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(versions[0])
    origins[str(target.relative_to(out))] = dict(git_commit=receipt['harness_source']['commit'], git_path=name)
for attempt in ['review-a', 'review-b', 'review-c']:
    for path in sorted((log / attempt).iterdir()):
        if path.suffix in ['.json', '.log']:
            copy(path, 'review/' + attempt + '/' + path.name)
for pattern in ['stub-*.log', 'exports-before.log', 'ci-python-docs-8d2bd26a.log']:
    for path in sorted(log.glob(pattern)):
        copy(path, 'review/' + path.name)
review = read(log / 'review-c/receipt.json')
assert review['status'] == 'passed' and 'active' not in review and len(review['steps']) == 32
assert all(step['exit_code'] == 0 for step in review['steps'])
for name, expected in review['products'].items():
    assert sha(log / 'review-c' / name) == expected
summary = dict(schema='spiraltorch.resident_direct_vjp_wide.results.v1', status='passed',
    harness_source=receipt['harness_source'], workers=receipt['workers'],
    steps=18, seconds=receipt['seconds'], optimizer_recipes=len(timings), retained_intervals=4320,
    browser_intervals_revalidated=2160, rows=rows, timings=timings,
    max_abs_error=max(t['max_abs_error'] for t in timings),
    products=receipt['products'], unchanged_helpers=receipt['unchanged_helpers'],
    boundary=receipt['boundary'], publication='results_validation_and_hashes_only',
    raw_payloads='retained locally; absent from Git',
    review_source=review['source'], review_steps=len(review['steps']),
    raw_inventory_sha256=sha(out / 'raw-inventory.json'))
write(out / 'summary.json', summary)
records = [dict(path=str(path.relative_to(out)), bytes=path.stat().st_size,
    sha256=sha(path), origin=origins.get(str(path.relative_to(out)), 'derived'))
    for path in sorted(out.rglob('*')) if path.is_file()]
write(out / 'manifest.json', dict(schema='spiraltorch.results_manifest.v1', records=records,
    published_bytes=sum(r['bytes'] for r in records)))
print(json.dumps(dict(status='packaged', published_files=len(records),
    published_bytes=sum(r['bytes'] for r in records), local_raw_bytes=sum(r['bytes'] for r in inventory))))
