"""Verify published result bytes and aggregation; optionally check local raw bytes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def member(root, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('invalid manifest member')
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('manifest member escapes root')
    return path


parser = argparse.ArgumentParser(__doc__)
parser.add_argument('root', nargs='?', type=Path, default=Path(__file__).resolve().parent)
parser.add_argument('--raw-root', type=Path)
args = parser.parse_args()
root = args.root.resolve()
manifest = read(root / 'manifest.json')
records = manifest['records']
assert len({r['path'] for r in records}) == len(records)
for record in records:
    path = member(root, record['path'])
    assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], record['path']
physical = {str(path.relative_to(root)) for path in root.rglob('*') if path.is_file()}
assert physical == {r['path'] for r in records} | {'manifest.json'}
assert sum(r['bytes'] for r in records) == manifest['published_bytes']
summary = read(root / 'summary.json')
receipt = read(root / 'run/receipt.json')
assert receipt['status'] == summary['status'] == 'passed' and 'active' not in receipt
assert len(receipt['steps']) == summary['steps'] == 18 and all(s['exit_code'] == 0 for s in receipt['steps'])
assert receipt['harness_source'] == summary['harness_source'] and receipt['workers'] == summary['workers']
assert sha(root / 'harness/measure.py') == receipt['harness_sha256']
assert sha(root / 'raw-inventory.json') == summary['raw_inventory_sha256']
for name, expected in receipt['unchanged_helpers'].items():
    assert sha(member(root / 'sources', name)) == expected
inventory = read(root / 'raw-inventory.json')
raw_hashes = {str(Path(inventory['root']) / r['path']): r['sha256'] for r in inventory['records']}
assert len(raw_hashes) == len(inventory['records'])
assert sum(r['bytes'] for r in inventory['records']) == inventory['bytes']
expected_timings, expected_rows = [], []
intervals = browser_intervals = 0
for repetition in ['a', 'b']:
    for mode in ['plain', 'topos_ema', 'clipped_topos_ema']:
        filename = repetition + '-' + mode + '-validation.json'
        path = root / 'run' / filename
        assert sha(path) == raw_hashes[str(Path(inventory['root']) / filename)]
        data = read(path)
        assert data['status'] == 'passed' and data['matrix'] == 'wide' and len(data['cases']) == 9
        assert data['learner_optimizer'] == (None if mode == 'plain' else mode)
        assert data['browser_intervals_revalidated'] == 360
        browser_intervals += data['browser_intervals_revalidated']
        for item in data['inputs']:
            assert item['sha256'] == raw_hashes[item['path']]
        for role, source in receipt['workers'].items():
            binding = data['source_bindings'][role]
            assert binding['valid'] and all(binding['checks'].values())
            assert binding['source_commit'] == source['commit'] and binding['source_tree'] == source['tree']
        for case in data['cases']:
            expected_timings.append(dict(run=repetition, mode=mode, config=case['config'],
                native=case['native'], browser=case['browser'], max_abs_error=max(case['max_abs_errors'].values())))
            for route in ['native', 'browser']:
                for cadence in ['immediate', 'deferred']:
                    result = case[route][cadence]
                    lanes = result['lanes']
                    assert all(lane['retained'] == 8 for lane in lanes.values())
                    intervals += sum(lane['retained'] for lane in lanes.values())
                    assert math.isclose(result['baseline_over_candidate'],
                        lanes['baseline']['median_ms'] / lanes['candidate']['median_ms'], rel_tol=1e-14)
                    if route == 'native':
                        assert math.isclose(result['torch_over_candidate'],
                            lanes['torch']['median_ms'] / lanes['candidate']['median_ms'], rel_tol=1e-14)
        for cadence in ['immediate', 'deferred']:
            row = dict(run=repetition, mode=mode, cadence=cadence)
            for route in ['native', 'browser']:
                cases = [c[route][cadence] for c in data['cases']]
                ratios = [c['baseline_over_candidate'] for c in cases]
                row[route] = dict(min=min(ratios), max=max(ratios), median_case_ratio=statistics.median(ratios),
                    geomean=math.exp(sum(math.log(r) for r in ratios) / len(ratios)),
                    candidate_faster=sum(r > 1 for r in ratios))
                if route == 'native':
                    torch = [c['torch_over_candidate'] for c in cases]
                    row[route].update(torch_faster=sum(r < 1 for r in torch),
                        torch_over_candidate_range=[min(torch), max(torch)])
            expected_rows.append(row)
assert summary['timings'] == expected_timings and summary['rows'] == expected_rows
assert len(expected_timings) == summary['optimizer_recipes'] == 54
assert intervals == summary['retained_intervals'] == 4320
assert browser_intervals == summary['browser_intervals_revalidated'] == 2160
assert summary['max_abs_error'] == max(t['max_abs_error'] for t in expected_timings)
raw_checked = 0
if args.raw_root is not None:
    for record in inventory['records']:
        path = member(args.raw_root, record['path'])
        assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], record['path']
        raw_checked += 1
print(json.dumps(dict(status='passed', manifest_sha256=sha(root / 'manifest.json'),
    published_files=len(records), optimizer_recipes=len(expected_timings), retained_intervals=intervals,
    local_raw_files_hash_checked=raw_checked, local_raw_bytes=inventory['bytes'] if raw_checked else None,
    numerical_replay_performed=False,
    boundary='Checks bytes, source bindings and all saved result aggregation; does not rerun numerical validation or GPU work.')))
