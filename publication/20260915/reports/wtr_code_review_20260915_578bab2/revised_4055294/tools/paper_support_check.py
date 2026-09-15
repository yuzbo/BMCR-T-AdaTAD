"""Run CPU contracts or the selected course's real two-update GPU preflight."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from h65.paper.runtime import read_config,json_write

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--output');a=p.parse_args();cfg=read_config(a.config)
    if a.device=='cuda':
        out=a.output or str(ROOT/'research/paper/runs'/('preflight_'+cfg['id']))
        raise SystemExit(subprocess.run([sys.executable,str(ROOT/'tools/paper_train.py'),'--config',a.config,'--preflight','--output',out],cwd=ROOT).returncode)
    import unittest
    sys.path.insert(0,str(ROOT/'upstream'))
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests/paper'));result=unittest.TextTestRunner(verbosity=2).run(suite)
    record=dict(config=cfg['id'],tests=result.testsRun,passed=result.wasSuccessful(),scope='CPU real-module micro-contracts; not GPU/pretrained full-course validation')
    if a.output:json_write(Path(a.output)/'cpu_contracts.json',record)
    print(json.dumps(record));raise SystemExit(not result.wasSuccessful())
