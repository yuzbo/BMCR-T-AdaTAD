"""Start the single-seed plan after the recorded old-seed cancellation."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];EXP=ROOT/'research/paper'
record=json.loads((EXP/'seed42_transition/cancellation.json').read_text())
assert record['required_seed']==42
subprocess.run([sys.executable,str(ROOT/'tools/paper_plan.py')],cwd=ROOT,check=True)
plan=json.loads((EXP/'plan.json').read_text())
assert len(plan['configs'])==52 and all(c['seed']==42 for c in plan['configs'])
assert len({(c['dataset'],c['backbone'],c['head'],c['comparison']) for c in plan['configs']})==52
assert all(d in plan['stages'] for s in plan['stages'].values() for d in s.get('dependencies',[]))
state=dict(recipe=plan['recipe'],stages=plan['stages'],required_seed=42,runs_per_configuration=1)
(EXP/'deployment.json').write_text(json.dumps(state,indent=2)+'\n')
with (EXP/'slurm/controller_seed42.log').open('a') as log:
    child=subprocess.Popen([sys.executable,'-u','tools/paper_dispatch.py','--submit','--max-live','10','--legacy-receipt',str(ROOT.parent/'fpw_3d_20260913/research/frame/deployment.json'),'--inherit-legacy'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(controller_pid=child.pid,configurations=52,stages=len(plan['stages']),seed=42)))
