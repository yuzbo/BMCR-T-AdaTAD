"""Retrieve existing Raw logs/JSON only; no reader or model execution."""
import json
import zipfile
from pathlib import Path
from deploy_probe import remote,SITES,OUT,LOCAL
body="""import json,time
from pathlib import Path
root=Path('/data/run01/sczc063/yuzibo/wtr_raw_v1_20260915')
folders=[root/'logs',root/'results',root/'mini_bank',root/'validation',
 root/'revision_27d557e/logs',root/'revision_27d557e/mini_bank',root/'revision_27d557e/results']
files={}
for folder in folders:
 if not folder.exists():continue
 for path in folder.rglob('*'):
  if path.is_file() and path.suffix in ('.json','.jsonl','.log','.md','.txt') and 'resources' not in path.name.lower():
   files[str(path.relative_to(root))]=path.read_text()
for name in ('revision_27d557e/runtime/protocol.json','CODE_REVISION','source_revision.txt'):
 path=root/name
 if path.exists():files[name]=path.read_text()
print(json.dumps(dict(recorded_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),root=str(root),
 root_children=[p.name for p in root.iterdir()],files=files)))
"""
result=json.loads(remote('4090',SITES['4090']['python']+' -',body.encode()))
target=OUT/'github_release_assets/raw-existing-evidence-and-logs.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as bundle:
    for name,content in result['files'].items():bundle.writestr(name,content)
    inventory={name:len(value.encode()) for name,value in result['files'].items()}
    bundle.writestr('CAPTURE.json',json.dumps(dict(recorded_at=result['recorded_at'],root=result['root'],files=inventory),indent=2))
hub=LOCAL/'research/wtr_rfv_release_20260915'
raw=json.loads((OUT.parent/'wtr_fasttrack_20260915/current_model_report/4090.json').read_text())['raw']
(hub/'raw/MEASURED_SNAPSHOT_1150.json').write_text(json.dumps(raw,indent=2),encoding='utf8')
receipt=dict(name=target.name,bytes=target.stat().st_size,files=len(inventory),recorded_at=result['recorded_at'],
    scope='Existing Raw JSON/log files; no new query, training or domain expansion',root_children=result['root_children'])
(OUT/'RAW_PUBLICATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
print(json.dumps(receipt))
