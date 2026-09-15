"""Read the one submitted A100 diagnostic and retrieve its completed artifacts."""
import json
from deploy_probe import remote,SITES,OUT
from deploy_autodl import ssh,PYTHON,ROOT
body="""import json,subprocess
from pathlib import Path
root=Path(ROOT);out=root/'results/7245972/DS_DIAGNOSTIC'
state=Path('/proc/317797/stat')
queue=state.read_text().split()[2] if state.exists() else 'EXITED'
files={p.name:p.read_text() for p in out.glob('*.json')} if out.exists() else {}
log=root/'logs/ds_diagnostic_7245972.log'
print(json.dumps(dict(queue=queue,account='AutoDL GPU1; pendingA100248477 moved',files=files,log=log.read_text() if log.exists() else '')))
"""
result=json.loads(ssh(PYTHON+' -',('ROOT='+repr(ROOT)+'\n'+body).encode()))
folder=OUT/'ds_diagnostic_7245972';folder.mkdir(exist_ok=True)
for name,value in result['files'].items():(folder/name).write_text(value,encoding='utf8')
(folder/'ds_diagnostic_7245972.log').write_text(result['log'],encoding='utf8')
(folder/'status.json').write_text(json.dumps({k:result[k] for k in ('queue','account')},indent=2),encoding='utf8')
print(json.dumps({k:result[k] for k in ('queue','account')}))
print('\n'.join(result['log'].splitlines()[-18:]))
