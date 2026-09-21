"""Load only the frozen current-source native product; never the installed wheel."""
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import unittest

library, workspace, mode, *args = sys.argv[1:]
library = Path(library).resolve()
workspace = Path(workspace).resolve()
name = 'spiraltorch.spiraltorch'
loader = importlib.machinery.ExtensionFileLoader(name, str(library))
spec = importlib.util.spec_from_file_location(name, library, loader=loader)
native = importlib.util.module_from_spec(spec)
sys.modules[name] = native
spec.loader.exec_module(native)
sys.path.insert(0, str(workspace/'bindings/st-py'))
import spiraltorch as st
assert Path(st._rs.__file__).resolve() == library
if mode == 'tests':
    suite = unittest.TestSuite()
    for filename in ('test_nn_resident.py','test_nn_resident_training.py','test_nn_resident_graph_training.py',
                     'test_wgpu_tensor.py','test_nn_resident_graph_forward.py'):
        spec = importlib.util.spec_from_file_location(Path(filename).stem, workspace/'bindings/st-py/tests'/filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(status='passed' if result.wasSuccessful() else 'failed', tests=result.testsRun,
                  skipped=len(result.skipped),native_path=str(library),build_info=st.build_info(),
                  wgpu_available=st.wgpu_kernel_reports_available())
    with Path(args[0]).open('x') as output: json.dump(report,output,indent=2)
    print(json.dumps(report))
    if not result.wasSuccessful(): raise SystemExit(1)
else:
    sys.argv=[mode,*args]
    runpy.run_path(str(workspace/mode),run_name='__main__')
