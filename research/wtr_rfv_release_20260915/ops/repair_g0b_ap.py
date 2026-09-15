"""Finish already-captured full-window predictions with the fixed CPU AP path."""
from pathlib import Path
from deploy_probe import package,OUT
from deploy_autodl import ssh,upload,PYTHON,ROOT

REV='dca6564e77e13717adf461dcf1feabf8bc18bbda'
CODE=ROOT+'/versions/'+REV
DATA=ROOT+'/results/2ca4d4b/T_LOCAL_CF_G0B'
archive=package(REV);ssh('mkdir -p '+CODE);upload(archive,CODE+'/source.tar.gz')
ssh('tar -xzf '+CODE+'/source.tar.gz -C '+CODE)
body='''import json,sys,subprocess,os,time
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
'''
header='\n'.join(key+'='+repr(value) for key,value in dict(CODE=CODE,DATA=DATA,ROOT=ROOT,REV=REV,PYTHON=PYTHON).items())+'\n'
remote_file=CODE+'/repair_g0b_once.py'
local=OUT/'repair_g0b_once.py';local.write_text(header+body,encoding='utf-8');upload(local,remote_file)
launch='''import json,os,subprocess,time
from pathlib import Path
root=Path(ROOT);receipt=root/'receipts/g0b_ap_repair.json'
if receipt.exists():print(receipt.read_text());raise SystemExit(0)
log=root/'logs/g0b_ap_repair.log';env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
with log.open('w') as stream:process=subprocess.Popen([PYTHON,SCRIPT],env=env,cwd=CODE,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
data=dict(pid=process.pid,postprocess_revision=REV,gpu_queries_repeated=0,log=str(log),started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
receipt.write_text(json.dumps(data,indent=2));print(json.dumps(data))
'''
head='\n'.join(key+'='+repr(value) for key,value in dict(ROOT=ROOT,REV=REV,PYTHON=PYTHON,SCRIPT=remote_file,CODE=CODE).items())+'\n'
result=ssh(PYTHON+' -',(head+launch).encode(),attempts=1)
(OUT/'g0b_repair_launch.json').write_text(result,encoding='utf-8');print(result)
