CODE='/root/autodl-tmp/rfv_20260915/versions/dca6564e77e13717adf461dcf1feabf8bc18bbda'
DATA='/root/autodl-tmp/rfv_20260915/results/2ca4d4b/T_LOCAL_CF_G0B'
ROOT='/root/autodl-tmp/rfv_20260915'
REV='dca6564e77e13717adf461dcf1feabf8bc18bbda'
PYTHON='/root/autodl-tmp/envs/opentad/bin/python'
import json,sys,subprocess,os,time
from pathlib import Path
from types import SimpleNamespace
code=Path(CODE);folder=Path(DATA);root=Path(ROOT)
sys.path[:0]=[str(code),str(code/'upstream')]
from h65.paper.runtime import build_config,json_write
from h65.atlas.data import CharacterizationData
from tools.rfv_run import finish_ap
meta=json.loads((folder/'manifest_0.json').read_text())
resources=json.loads((root/'versions/2ca4d4b5f2ad657410bfc5c2c4cebd3bc6a11a32/runtime/resources.json').read_text())
cfg=build_config(meta['config'],resources)
source=CharacterizationData(cfg,resources,'development',meta['video_ids'],one_window=False)
receipts=[json.loads(path.read_text()) for path in (folder/'windows').glob('*.json')]
if len(receipts)!=len(source) or {x['video_id'] for x in receipts}!=set(meta['video_ids']):raise RuntimeError('Saved full-video predictions are incomplete')
finish_ap(SimpleNamespace(model_cfg=cfg,identity=meta['checkpoint']),source,folder,SimpleNamespace(shards=1,mode='local_cf'))
json_write(folder/'completed_0.json',dict(source_revision=meta['source_revision'],postprocess_revision=REV,
 video_ids=meta['video_ids'],group_files=sorted({item for row in receipts for item in row['group_files']}),
 windows=len(source),checkpoint=meta['checkpoint'],mode='local_cf',gpu_queries_repeated=0))
print(json.dumps(dict(repaired=True,windows=len(source),postprocess_revision=REV,gpu_queries_repeated=0)),flush=True)
subprocess.run([PYTHON,str(code/'tools/rfv_analyze.py'),'headroom','--capture',str(folder),
 '--output',str(folder.parent/'analysis_g0b')],check=True)
