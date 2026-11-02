import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('admission', Path(__file__).with_name('admit_case24_once.py'))
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)

class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.env = {'GITHUB_REPOSITORY': a.REPO, 'GITHUB_REF': 'refs/heads/' + a.BRANCH,
                    'GITHUB_EVENT_NAME': 'push', 'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_RUN_ID': '7',
                    'GITHUB_SHA': 'new-head', 'JOB_START_EPOCH': '123'}
        self.event = {'before': a.SOURCE, 'after': 'new-head', 'head_commit': {'message': a.MESSAGE}}
        self.history = {'total_count': 1, 'workflow_runs': [{'id': 7, 'head_sha': 'new-head', 'run_attempt': 1}]}
    def test_duplicate_and_manual_reruns_rejected_before_git(self):
        for field, value in [('GITHUB_RUN_ATTEMPT','2'), ('GITHUB_REF','refs/heads/main'), ('GITHUB_EVENT_NAME','workflow_dispatch')]:
            with self.subTest(field=field), patch.object(a.subprocess, 'check_output') as git:
                with self.assertRaises(ValueError): a.admit(Path('/nonexistent'), dict(self.env, **{field:value}), self.event, self.history)
                git.assert_not_called()
        self.history['total_count'] = 2
        with patch.object(a.subprocess,'check_output') as git:
            with self.assertRaises(ValueError): a.admit(Path('/nonexistent'), self.env, self.event, self.history)
            git.assert_not_called()
    def test_exact_manifest_and_path_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'tools').mkdir(); (root/'probe.rs').write_text('unchanged arithmetic')
            manifest = {'source_sha':a.SOURCE, 'files':{'probe.rs':a.hashlib.sha256((root/'probe.rs').read_bytes()).hexdigest()}}
            raw = json.dumps(manifest).encode(); (root/a.MANIFEST).write_bytes(raw)
            self.env['ADMISSION_SHA256'] = a.hashlib.sha256(raw).hexdigest()
            def git(cmd, **kw):
                return {'rev-parse HEAD':'new-head','rev-parse HEAD^':a.SOURCE,'rev-parse HEAD^{tree}':'tree',
                        'status --porcelain':'','diff --name-only '+a.SOURCE+' HEAD':'\n'.join(['probe.rs',a.MANIFEST,a.WORKFLOW])}[' '.join(cmd[1:])]
            with patch.object(a.subprocess,'check_output',side_effect=git):
                self.assertEqual(a.admit(root,self.env,self.event,self.history)['gpu_invocations_allowed'],1)
                (root/'probe.rs').write_text('changed')
                with self.assertRaisesRegex(ValueError,'file identity changed'): a.admit(root,self.env,self.event,self.history)

if __name__ == '__main__': unittest.main()
