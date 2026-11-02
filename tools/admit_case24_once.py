#!/usr/bin/env python3
"""Read-only hosted admission; reject any repeated authorization slot before GPU."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

SOURCE = 'e08b8a527f64826a499ce129879a13fa0e6f3b33'
REPO = 'RyoSpiralArchitect/SpiralTorch'
BRANCH = 'spiralreality/case24-diagnostic-20261102-once'
WORKFLOW = '.github/workflows/case24_once_20261102.yml'
MANIFEST = 'tools/case24_once_20261102.json'
MESSAGE = 'diagnostic: one authorized case24 invocation 20261102-6855f1c8'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def admit(root, env, event, history):
    require(env['GITHUB_REPOSITORY'] == REPO, 'wrong repository')
    require(env['GITHUB_REF'] == 'refs/heads/' + BRANCH, 'wrong branch')
    require(env['GITHUB_EVENT_NAME'] == 'push', 'only original push is admitted')
    require(env['GITHUB_RUN_ATTEMPT'] == '1', 'reruns forbidden')
    require(event['before'] == SOURCE and event['after'] == env['GITHUB_SHA'], 'unexpected ref transition')
    require(event['head_commit']['message'] == MESSAGE, 'wrong authorization slot')
    runs = history['workflow_runs']
    require(history['total_count'] == 1 and len(runs) == 1, 'authorization slot already has another run')
    require(str(runs[0]['id']) == env['GITHUB_RUN_ID'], 'history not this invocation')
    require(runs[0]['head_sha'] == env['GITHUB_SHA'] and runs[0]['run_attempt'] == 1, 'history owner changed')
    git = lambda *args: subprocess.check_output(['git', *args], cwd=root, text=True).strip()
    require(git('rev-parse', 'HEAD') == env['GITHUB_SHA'], 'checkout not trigger head')
    require(git('rev-parse', 'HEAD^') == SOURCE, 'source parent changed')
    require(not git('status', '--porcelain'), 'checkout is not clean')
    data = (root / MANIFEST).read_bytes()
    require(hashlib.sha256(data).hexdigest() == env['ADMISSION_SHA256'], 'admission manifest changed')
    manifest = json.loads(data)
    require(manifest['source_sha'] == SOURCE, 'manifest source changed')
    changed = set(git('diff', '--name-only', SOURCE, 'HEAD').splitlines())
    require(changed == set(manifest['files']) | {WORKFLOW, MANIFEST}, 'unexpected changed paths')
    for name, expected in manifest['files'].items():
        require(hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, 'file identity changed: ' + name)
    return {'schema': 'spiraltorch.case24_once_admission.v1', 'source_sha': SOURCE,
            'instrumented_sha': env['GITHUB_SHA'], 'instrumented_tree': git('rev-parse', 'HEAD^{tree}'),
            'run_id': env['GITHUB_RUN_ID'], 'run_attempt': 1, 'branch': BRANCH,
            'admission_sha256': env['ADMISSION_SHA256'], 'user_authorization': 'Sentinel_6855f1c821a08191b89170262fb6831a',
            'job_start_epoch': env['JOB_START_EPOCH'], 'max_job_seconds': 5400,
            'max_process_deadline_offset_seconds': 5100, 'gpu_invocations_allowed': 1,
            'trajectories': ['continuous_updates_1_2', 'original_split_resume_update_2'],
            'case': 'mixture-11-3-train-query-frozen', 'release_ready': False}


def main():
    output = Path(sys.argv[1])
    # Reserve once even when admission fails; never overwrite or silently retry.
    with output.open('x') as stream:
        try:
            event = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_bytes())
            url = ('https://api.github.com/repos/' + REPO + '/actions/workflows/' + Path(WORKFLOW).name
                   + '/runs?branch=' + BRANCH + '&event=push&per_page=100')
            request = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + os.environ['GH_TOKEN'],
                'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28'})
            with urllib.request.urlopen(request, timeout=30) as response:
                history = json.load(response)
            receipt = admit(Path(__file__).resolve().parents[1], os.environ, event, history)
        except Exception as error:
            json.dump({'status': 'rejected', 'error': str(error), 'gpu_started': False}, stream, indent=2)
            raise
        json.dump(receipt, stream, indent=2)


if __name__ == '__main__':
    main()
