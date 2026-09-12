"""Two complete wide learner matrices; unchanged frozen workers and helpers."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/Users/ryospiralarchitect/\U0001f300SpiralReality\U0001f300/_wt/spiraltorch-resident-graph-forward-v1')
LOG = Path(__file__).parent
BASE = Path('/Users/ryospiralarchitect/Library/Logs/SpiralTorch/resident-capture-pool-20260912/candidate-a')
CANDIDATE = Path('/Users/ryospiralarchitect/Library/Logs/SpiralTorch/resident-direct-vjp-20260912/candidate-a')
PRIOR = Path('/Users/ryospiralarchitect/Library/Logs/SpiralTorch/resident-direct-vjp-20260912')
P = '/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12'
TORCH = '/Users/ryospiralarchitect/Library/Logs/SpiralTorch/rank-finite-bound-20260907/venv/bin/python'
CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
if len(sys.argv) == 3 and sys.argv[1] == '--launch':
    name = sys.argv[2]
    assert Path(name).name == name and not (LOG/name).exists()
    with (LOG/(name+'.log')).open('xb') as output:
        child = subprocess.Popen([P, '-S', str(Path(__file__).resolve()), name], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps(dict(pid=child.pid, log=str(LOG/(name+'.log')))))
    raise SystemExit(0)
assert len(sys.argv) == 2 and Path(sys.argv[1]).name == sys.argv[1]
OUT = LOG/sys.argv[1]
OUT.mkdir()
git = lambda *args: subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

assert not git('status', '--porcelain')
source = dict(commit=git('rev-parse', 'HEAD'), tree=git('rev-parse', 'HEAD^{tree}'))
products, workers = {}, {}
for role, folder in [('baseline', BASE), ('candidate', CANDIDATE)]:
    receipt = json.loads((folder/'receipt.json').read_bytes())
    assert receipt['status'] == 'passed' and 'active' not in receipt
    workers[role] = receipt['source']
    products[str(folder/'receipt.json')] = sha(folder/'receipt.json')
    for name, expected in receipt['products'].items():
        path = folder/name
        assert sha(path) == expected
        products[str(path)] = expected
for name in ['verified-a', 'measurement-a', 'measurement-b']:
    receipt = json.loads((PRIOR/name/'receipt.json').read_bytes())
    assert receipt['status'] == 'passed' and 'active' not in receipt
    assert receipt['source'] == workers['candidate']
    products[str(PRIOR/name/'receipt.json')] = sha(PRIOR/name/'receipt.json')
helpers = ['tools/bench_resident_training_vs_torch.py', 'tools/bench_resident_training_browser.cjs',
    'tools/validate_resident_training_bench.py', 'tools/resident_learner_bench_reference.py',
    'bindings/st-wasm/tests/resident_training_bench.html',
    'crates/st-nn/examples/support/resident_training_bench.rs',
    'crates/st-nn/examples/support/resident_learner_bench.rs']
identical = {}
for path in helpers:
    versions = [subprocess.check_output(['git','show',ref+':'+path],cwd=ROOT)
        for ref in [workers['baseline']['commit'], workers['candidate']['commit'], source['commit']]]
    assert versions[0] == versions[1] == versions[2]
    identical[path] = hashlib.sha256(versions[0]).hexdigest()
env = dict(os.environ, SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS='1', PYTORCH_ENABLE_MPS_FALLBACK='0',
    NODE_PATH='/Users/ryospiralarchitect/Library/Logs/SpiralTorch/midk-seek-20260906/browser-deps/node_modules')
report = dict(status='running', pid=os.getpid(), harness_source=source, workers=workers, steps=[],
    products=products, unchanged_helpers=identical, harness_sha256=sha(Path(__file__)),
    orders=[['plain','topos_ema','clipped_topos_ema'],['clipped_topos_ema','topos_ema','plain']],
    boundary='Unchanged wide recipes: [4,64,64]/depth8, [2,128,128]/depth8, [2,64,256]/depth4; '
    'seeds 17/29/43, two warmups/eight retained intervals, eight updates each, immediate/deferred acceptance. '
    'Two full runs, reversed optimizer order only; all numerical results and timings retained. '
    'Native Metal and browser WebGPU versus eager Torch MPS, no CPU fallback. '
    'This is a separate size stratum, not a replacement for earlier mixed/slower standard results. '
    'Host exclusivity UNKNOWN; owned GPU work serial. No phase attribution, fastest-Torch or FT-quality claim.')

def save():
    path = OUT/'receipt.json.tmp'
    path.write_text(json.dumps(report, indent=2)+'\n')
    path.replace(OUT/'receipt.json')

def run(name, args):
    args = list(map(str, args))
    started = time.monotonic()
    print('START', name, flush=True)
    with (OUT/(name+'.log')).open('xb') as log:
        child = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        report['active'] = dict(name=name, pid=child.pid, command=args)
        save()
        code = child.wait()
    report.pop('active')
    report['steps'].append(dict(name=name, command=args, exit_code=code, seconds=time.monotonic()-started))
    save()
    print('END', name, code, flush=True)
    if code: raise RuntimeError(name)
    assert not git('status', '--porcelain') and git('rev-parse', 'HEAD') == source['commit']

start = time.monotonic()
try:
    baseline_ref, candidate_ref = workers['baseline']['commit'], workers['candidate']['commit']
    for index, modes in enumerate(report['orders']):
        for mode in modes:
            prefix = ('a' if index == 0 else 'b')+'-'+mode
            optimizer = [] if mode == 'plain' else ['--learner-optimizer', mode]
            native, browser = OUT/(prefix+'-native.json'), OUT/(prefix+'-browser.json')
            run(prefix+'-native', [TORCH, '-I', ROOT/'tools/bench_resident_training_vs_torch.py',
                '--baseline', BASE/'resident_training_bench', '--baseline-source', baseline_ref,
                '--candidate', CANDIDATE/'resident_training_bench', '--candidate-source', candidate_ref,
                '--device', 'mps', '--graph', '--learner', '--matrix', 'wide', *optimizer, '--output', native])
            run(prefix+'-browser', ['node', ROOT/'tools/bench_resident_training_browser.cjs',
                BASE/'wasm', CANDIDATE/'wasm', CHROME, browser,
                'learner', 'wide', 'none', 'none' if mode == 'plain' else mode])
            run(prefix+'-validate', [P, '-I', ROOT/'tools/validate_resident_training_bench.py',
                '--native', native, '--browser', browser, '--browser-progress', str(browser)+'.progress.jsonl',
                '--baseline-source', baseline_ref, '--candidate-source', candidate_ref,
                '--browser-harness-source', source['commit'], '--output', OUT/(prefix+'-validation.json')])
            for path in [native, browser, OUT/(prefix+'-validation.json')]:
                value = json.loads(path.read_bytes())
                assert value['status'] == 'passed' and len(value['cases']) == 9 and value['matrix'] == 'wide'
                report['products'][str(path)] = sha(path)
    for path, expected in report['products'].items(): assert sha(Path(path)) == expected
    assert len(report['steps']) == 18
    report['status'] = 'passed'
except BaseException as error:
    report.update(status='error', error=repr(error))
    raise
finally:
    report['seconds'] = time.monotonic()-start
    save()
