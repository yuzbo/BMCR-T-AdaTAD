"""Replace only publisher-mismatched shards, preserve originals, then resume preparation."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research/paper/recovery_20260914_0240'
DOWNLOADS=Path('/data/run01/sczc063/yuzibo/bcr_tad_v3_implementation/activitynet/downloads')
def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use the owned preparation allocation')
    audit=json.loads((OUT/'validation_shards.json').read_text())
    if not audit['complete']:raise RuntimeError('Validation shard scan is incomplete')
    repaired=[]
    for row in audit['rows']:
        if row['matches']:continue
        original=Path(row['path']).resolve()
        if original.parent!=DOWNLOADS.resolve():raise ValueError('Unexpected shard directory')
        incoming=DOWNLOADS/'repair_downloads_1288466'/original.name
        url='https://huggingface.co/datasets/YimuWang/ActivityNet/resolve/main/'+original.name+'?download=true'
        print(json.dumps(dict(action='verify_desktop_relayed_shard',path=str(original),incoming=str(incoming),url=url)),flush=True)
        if not incoming.exists() or incoming.stat().st_size!=row['bytes']:raise RuntimeError('Desktop HTTPS relay has not completed; compute nodes have no HTTPS route')
        digest=hashlib.sha256()
        with incoming.open('rb') as stream:
            while chunk:=stream.read(8*1024*1024):digest.update(chunk)
        if incoming.stat().st_size!=row['bytes'] or digest.hexdigest()!=row['expected']:
            raise RuntimeError('Replacement does not match the publisher; original preserved')
        backup=DOWNLOADS/'corrupt_1288466';backup.mkdir(exist_ok=True)
        target=backup/original.name
        if target.exists():raise RuntimeError('Previous repair backup exists; inspect before repeating')
        original.rename(target);incoming.rename(original)
        repaired.append(dict(path=str(original),original_preserved=str(target),publisher_match=True))
        (OUT/'shard_repair.json').write_text(json.dumps(dict(job_id=os.environ['SLURM_JOB_ID'],repaired=repaired,time=time.strftime('%Y-%m-%dT%H:%M:%S%z')),indent=2)+'\n')
    return subprocess.run([sys.executable,'-u',str(ROOT/'tools/paper_data/prepare_anet_archives.py'),
                           '--ready-output',str(ROOT/'research/paper/assets/anet_ready.json')],cwd=ROOT).returncode
if __name__=='__main__':raise SystemExit(main())
