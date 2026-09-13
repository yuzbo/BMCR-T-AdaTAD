#!/usr/bin/env python3
"""Run synthetic CPU checks and write a machine-readable receipt."""
from pathlib import Path
import sys, unittest, json, platform, io
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import torch
from reference.core import dense_reference_macs
stream=io.StringIO()
suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_reference.py')
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
receipt=dict(evidence_kind='synthetic_cpu_mechanism_test',project_checkpoint_loaded=False,
             gpu_used=False,project_source_executed=False,seed=3407,python=platform.python_version(),
             torch=torch.__version__,tests_run=result.testsRun,failures=len(result.failures),
             errors=len(result.errors),success=result.wasSuccessful(),log=stream.getvalue(),
             estimates={b:dict(components_macs=dense_reference_macs(c),
                              total_macs=sum(dense_reference_macs(c).values())) for b,c in [('S',384),('B',768)]})
(ROOT/'validation').mkdir(exist_ok=True)
(ROOT/'validation/reference_checks.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(stream.getvalue()); sys.exit(0 if result.wasSuccessful() else 1)
