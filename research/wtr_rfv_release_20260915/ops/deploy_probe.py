"""Deploy an isolated RFV source snapshot to the existing two SSH clusters."""
import argparse
import concurrent.futures
import json
import io
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

LOCAL=Path('C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/rfv_sprint_20260915')
OUT=Path(__file__).resolve().parent
SSH_CONFIG='C:/Users/skywalker/Documents/ChatGPT/refine-tad/V3/motivation/ssh_config'
SITES={
    '4090':dict(host='bcr-4090',root='/data/run01/sczc063/yuzibo/rfv_20260915',
        python='/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python',
        resources='/data/run01/sczc063/yuzibo/wtr_fasttrack_20260915/research/paper/resources.local.json'),
    'a100':dict(host='bcr-a100',root='/HOME/pxyai/pxyai_0057/HDD_POOL/yzb/rfv_20260915',
        python='/HOME/pxyai/pxyai_0057/HDD_POOL/yzb/geosparse_tad_20260907/envs/opentad/bin/python',
        resources='/HOME/pxyai/pxyai_0057/HDD_POOL/yzb/wtr_fasttrack_20260915/research/paper/resources.local.json')}


def remote(site,command,stdin=None):
    options=['-F',SSH_CONFIG,'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3']
    for attempt in range(3):
        result=subprocess.run(['ssh',*options,SITES[site]['host'],command],input=stdin,capture_output=True)
        if result.returncode==0:return result.stdout.decode('utf-8',errors='replace')
        if result.returncode!=255:
            raise RuntimeError(site+': '+(result.stdout+result.stderr).decode('utf-8',errors='replace')[-8000:])
        time.sleep(2)
    raise RuntimeError(site+': '+result.stderr.decode('utf-8',errors='replace'))


def package(ref=None):
    ref=ref or subprocess.check_output(['git','rev-parse','HEAD'],cwd=LOCAL,text=True).strip()
    path=OUT/('rfv_source_'+ref[:7]+'.tar.gz')
    raw=subprocess.check_output(['git','archive','--format=tar',ref,'--','h65','tools','tests','configs','upstream','references','research/rfv_sprint_20260915'],cwd=LOCAL)
    with tarfile.open(path,'w:gz',compresslevel=2) as archive:
        for name in ('WTR_RFV_SCIENCE_SHA','source_revision.txt','EVALUATION_REVISION'):
            data=(ref+'\n').encode();info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644
            archive.addfile(info,io.BytesIO(data))
        with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as original:
            for member in original:
                if member.isfile() and Path(member.name).suffix not in {'.pyc','.pth','.pt'}:
                    archive.addfile(member,original.extractfile(member))
    return path


def deploy(site,archive,label,test,test_file):
    cfg=SITES[site];target=cfg['root']+'/'+label
    remote(site,'mkdir -p '+target)
    options=['-F',SSH_CONFIG,'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3']
    for attempt in range(3):
        copy=subprocess.run(['scp',*options,str(archive),cfg['host']+':'+target+'/source.tar.gz'],capture_output=True)
        if copy.returncode==0:break
        time.sleep(2)
    else:raise RuntimeError(site+': source upload failed')
    remote(site,'tar -xzf '+target+'/source.tar.gz -C '+target)
    output=remote(site,'cd '+target+' && '+cfg['python']+' tools/rfv_run.py prepare --resources '+cfg['resources']+' --output runtime')
    (OUT/('prepare_'+site+'.txt')).write_text(output,encoding='utf-8')
    print(json.dumps(dict(site=site,root=target,prepared=True)),flush=True)
    if test:
        result=remote(site,'cd '+target+' && PYTHONPATH='+target+':'+target+'/upstream OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 '+cfg['python']+' -m pytest -q '+test_file)
        (OUT/('cpu_'+site+'.txt')).write_text(result,encoding='utf-8')
        print(json.dumps(dict(site=site,cpu_test=result[-3000:])),flush=True)
    return dict(site=site,root=target,python=cfg['python'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True)
    parser.add_argument('--sites',nargs='+',choices=SITES,default=list(SITES));parser.add_argument('--test',action='store_true')
    parser.add_argument('--test-file',default='tests/test_rfv_contracts.py')
    args=parser.parse_args();archive=package(args.label.split('/',1)[1] if args.label.startswith('versions/') else None)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.sites)) as pool:
        results=list(pool.map(lambda site:deploy(site,archive,args.label,args.test,args.test_file),args.sites))
    (OUT/'deployment.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':main()
