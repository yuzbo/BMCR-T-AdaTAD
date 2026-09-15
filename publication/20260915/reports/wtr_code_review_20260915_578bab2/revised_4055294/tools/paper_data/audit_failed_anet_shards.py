"""Locate a real corrupt validation shard without re-downloading all 57 GB."""
import hashlib,json,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
WORK=Path('/data/run01/sczc063/yuzibo/bcr_tad_v3_implementation/OpenTAD')
DATA=WORK.parent/'activitynet';OUT=ROOT/'research/paper/recovery_20260914_0240'
def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use the owned data diagnostic allocation')
    OUT.mkdir(exist_ok=True)
    listing=json.loads((WORK/'reports/data/anet_source_research/yimuwang_tree.json').read_text())
    results=[]
    for item in sorted((x for x in listing if x['path'].startswith('v1-2_val.tar.gz.')),key=lambda x:x['path']):
        path=DATA/'downloads'/item['path'];digest=hashlib.sha256();begin=time.monotonic()
        with path.open('rb') as stream:
            while chunk:=stream.read(8*1024*1024):digest.update(chunk)
        actual=digest.hexdigest();expected=item['lfs']['oid']
        row=dict(path=str(path),expected=expected,actual=actual,matches=actual==expected,bytes=path.stat().st_size,seconds=time.monotonic()-begin)
        results.append(row);print(json.dumps(row),flush=True)
        (OUT/'validation_shards.json').write_text(json.dumps(dict(job_id=os.environ['SLURM_JOB_ID'],complete=False,rows=results),indent=2)+'\n')
    (OUT/'validation_shards.json').write_text(json.dumps(dict(job_id=os.environ['SLURM_JOB_ID'],complete=True,rows=results,repair_paths=[x['path'] for x in results if not x['matches']]),indent=2)+'\n')
if __name__=='__main__':main()
