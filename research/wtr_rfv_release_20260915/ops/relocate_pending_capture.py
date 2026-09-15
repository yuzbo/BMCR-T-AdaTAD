"""Replace only our not-yet-started RFV submissions after the GPU1 handoff."""
import json
from deploy_probe import remote,OUT

for site,job in [('4090',1290788),('a100',248292)]:
    code="""import json,subprocess
job=JOB
query=subprocess.run(['squeue','-j',str(job),'-h','-o','%T|%j'],capture_output=True,text=True,check=True).stdout.strip()
row=dict(job_id=job,observed=query,cancelled=False)
if query.startswith('PENDING|rfv-'):
 subprocess.run(['scancel',str(job)],check=True)
 row['cancelled']=True;row['reason']='replaced by owner-assigned AutoDL GPU1 to avoid Priority queue and duplicate CF capture'
print(json.dumps(row))
""".replace('JOB',str(job))
    python=('/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python' if site=='4090' else
        '/HOME/pxyai/pxyai_0057/HDD_POOL/yzb/geosparse_tad_20260907/envs/opentad/bin/python')
    result=json.loads(remote(site,python+' -',code.encode()))
    (OUT/('relocation_'+site+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(site=site,**result)),flush=True)
