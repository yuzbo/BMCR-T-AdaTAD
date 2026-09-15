"""Read current frozen reverse/B0 outputs; never launch or resume a process."""
import argparse
import json
from deploy_autodl import ssh,PYTHON,ROOT
from deploy_probe import OUT

parser=argparse.ArgumentParser();parser.add_argument('--revision',default='800bcd1');args=parser.parse_args()
REV=args.revision
body="""import json
from pathlib import Path
root=Path(ROOT);result=root/'results'/REV
files={}
for sub in ('REVERSE_CONTRACT','REVERSE_MINI','RISE_CHOICES'):
 directory=result/sub
 if directory.exists():
  for path in directory.iterdir():
   if path.suffix in ('.json','.jsonl','.txt'):files[sub+'/'+path.name]=path.read_text()
for name in ('sprint_progress.json','sprint_finished.json','run_receipt.json'):
 path=result/name
 if path.exists():files[name]=path.read_text()
log=root/'logs'/(('reverse_' if REV=='800bcd1' else 'reverse_fit_')+REV+'.log')
print(json.dumps(dict(files=files,log_tail=log.read_text().splitlines()[-40:] if log.exists() else [])))
"""
payload='ROOT='+repr(ROOT)+'\nREV='+repr(REV)+'\n'+body
result=json.loads(ssh(PYTHON+' -',payload.encode()))
folder=OUT/('reverse_'+REV);folder.mkdir(exist_ok=True)
for name,content in result['files'].items():
    path=folder/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content,encoding='utf8')
    if name.endswith('/REVERSE_CONTRACT.json'):
        data=json.loads(content)
        print(json.dumps(dict(contract=data['status'],records=[{k:r[k] for k in ('state_key','replay_max_error','cached_gain_error','gain_sum_max','reverse_descriptor_error','normalized_reverse_error','passed')} for r in data['records']])))
    if name.endswith('/RISE_CHOICES.json'):
        data=json.loads(content);print(json.dumps(dict(rise=data['summary'],legacy_checks=data['legacy_metric_checks'])))
    if name.endswith('/REVERSE_MINI.json'):
        data=json.loads(content);print(json.dumps(dict(mini=data['status'],held8={k:v['mean'] for k,v in data['aggregate']['held8'].items()},p2_minus_p1_bi=data['comparisons']['p2_minus_p1_bi']['regret'])))
    if name.endswith('progress.json') or name.endswith('finished.json'):print(content)
(folder/'log_tail.txt').write_text('\n'.join(result['log_tail'])+'\n',encoding='utf8')
print('\n'.join(result['log_tail'][-16:]))
