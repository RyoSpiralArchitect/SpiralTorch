"""Source-bound publication checks, run only after throughput measurements stop."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

LOG = Path(__file__).parent
ROOT = Path('/Users/ryospiralarchitect/\U0001f300SpiralReality\U0001f300/_wt/spiraltorch-resident-nn-integration-v1')
TARGET = Path('/Users/ryospiralarchitect/\U0001f300SpiralReality\U0001f300/_wt/spiraltorch-spiralk-golden-blackcat-contract-v1/target')
P = '/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12'
LOADER = '/Users/ryospiralarchitect/Library/Logs/SpiralTorch/resident-graph-forward-clients-20260910/python-client.py'
if len(sys.argv) == 3 and sys.argv[1] == '--launch':
    name = sys.argv[2]
    assert Path(name).name == name and not (LOG / name).exists()
    with (LOG / (name + '.log')).open('xb') as out:
        child = subprocess.Popen([P, '-S', str(Path(__file__).resolve()), name], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps(dict(pid=child.pid, log=str(LOG / (name + '.log')))))
    raise SystemExit(0)

assert len(sys.argv) == 2 and Path(sys.argv[1]).name == sys.argv[1]
wide = json.loads((LOG / 'measurement-a/receipt.json').read_bytes())
assert wide['status'] == 'passed' and 'active' not in wide
OUT = LOG / sys.argv[1]
OUT.mkdir()
git = lambda *args: subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
assert not git('status', '--porcelain')
source = dict(commit=git('rev-parse', 'HEAD'), tree=git('rev-parse', 'HEAD^{tree}'))
report = dict(status='running', source=source, pid=os.getpid(), steps=[], products={})
env = dict(os.environ, CARGO_BUILD_JOBS='4', CARGO_TARGET_DIR=str(TARGET),
    PYTHONNOUSERSITE='1', PYTORCH_ENABLE_MPS_FALLBACK='0')
env.pop('SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS', None)
start = time.monotonic()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save():
    temporary = OUT / 'receipt.json.tmp'
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(OUT / 'receipt.json')


def run(name, command, gpu=False):
    command = list(map(str, command))
    active_env = dict(env)
    if gpu:
        active_env['SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS'] = '1'
    began = time.monotonic()
    print('START', name, flush=True)
    with (OUT / (name + '.log')).open('xb') as output:
        child = subprocess.Popen(command, cwd=ROOT, env=active_env, stdout=output, stderr=subprocess.STDOUT)
        report['active'] = dict(name=name, pid=child.pid, command=command)
        save()
        code = child.wait()
    report.pop('active')
    report['steps'].append(dict(name=name, command=command, exit_code=code, seconds=time.monotonic() - began, gpu_opt_in=gpu))
    save()
    print('END', name, code, flush=True)
    if code:
        raise RuntimeError(name)
    assert not git('status', '--porcelain') and git('rev-parse', 'HEAD') == source['commit']


try:
    run('format', ['cargo', '+nightly-2026-04-15', 'fmt', '--all', '--', '--check'])
    run('diff-check', ['git', 'diff', '--check', 'origin/main...HEAD'])
    run('contracts', ['cargo', '+1.98.0', 'test', '--locked', '-p', 'st-kernel-contracts'])
    run('contracts-clippy', ['cargo', '+1.98.0', 'clippy', '--locked', '-p', 'st-kernel-contracts', '--all-targets', '--', '-D', 'warnings'])
    run('backend-clippy', ['cargo', '+1.98.0', 'clippy', '--locked', '-p', 'st-backend-wgpu', '--all-targets', '--', '-D', 'warnings'])
    run('tensor-clippy', ['cargo', '+1.98.0', 'clippy', '--locked', '-p', 'st-tensor', '--all-targets', '--no-default-features', '--features', 'cpu,wgpu', '--', '-D', 'warnings'])
    run('backend-guard-gpu', ['cargo', '+1.98.0', 'test', '--locked', '--release', '-p', 'st-backend-wgpu', 'resident_graph::guard_tests', '--', '--test-threads=1'], gpu=True)
    run('nn-ci-gpu', ['cargo', '+1.98.0', 'test', '--locked', '-p', 'st-nn', '--features', 'wgpu', '--release', '--lib', '--', '--test-threads=1'], gpu=True)
    tests = ['test_nn_resident_exports.py', 'test_nn_resident_graph_autograd.py',
        'test_nn_resident_graph_learner.py', 'test_nn_module_resident_forward.py',
        'test_nn_resident_module_update.py', 'test_nn_resident_loss.py',
        'test_nn_resident_classification.py', 'test_nn_resident_microbatch.py', 'test_wgpu_pointwise.py',
        'test_native_binding_ownership.py', 'test_unittest_smoke.py']
    for mode in ['gpu', 'cpu']:
        args = [] if mode == 'gpu' else ['--no-default-features', '--features', 'python-default']
        run('python-' + mode + '-build', ['cargo', '+1.98.0', 'build', '--locked', '--release', '-p', 'spiraltorch-py', *args])
        library = OUT / ('python-' + mode + '.dylib')
        shutil.copyfile(TARGET / 'release/libspiraltorch.dylib', library)
        report['products'][library.name] = sha(library)
        save()
        for test in tests:
            run(mode + '-' + Path(test).stem, [P, '-I', LOADER, library, ROOT, 'bindings/st-py/tests/' + test], gpu=mode == 'gpu')
    report['status'] = 'passed'
except BaseException as error:
    report.update(status='error', error=repr(error))
    raise
finally:
    report['seconds'] = time.monotonic() - start
    save()
