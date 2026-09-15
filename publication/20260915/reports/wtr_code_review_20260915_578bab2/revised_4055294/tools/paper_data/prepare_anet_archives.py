"""Consume v1.2 shards with bounded temporary storage, then finish full coverage.

The local raw/ZIP preparation must finish before this job starts. Archives stay
available for retry until every required video has a successful preparation receipt.
"""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import closing
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
from types import SimpleNamespace
import argparse
import os
import time

from prepare_anet_videos import convert, main as finish_preparation, video_id, is_video


WORK = Path('/data/run01/sczc063/yuzibo/bcr_tad_v3_implementation/OpenTAD')
DATA = Path('/data/run01/sczc063/yuzibo/datasets/activitynet')
STAGING = WORK.parent / 'activitynet'
ENV = Path('/data/run01/sczc063/yuzibo/conda_envs/opentad/bin')
LISTING = WORK / 'reports/data/anet_source_research/yimuwang_tree.json'


class ShardReader:
    def __init__(self, paths):
        self.paths = iter(paths)
        self.current = None

    def read(self, size):
        chunks = []
        remaining = size
        while remaining:
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.current = path.open('rb')
            chunk = self.current.read(remaining)
            if not chunk:
                self.current.close()
                self.current = None
            else:
                chunks.append(chunk)
                remaining -= len(chunk)
        return b''.join(chunks)

    def close(self):
        if self.current is not None:
            self.current.close()


def main(ready_output=None, workers=4):
    args = SimpleNamespace(annotation=str(DATA/'annotations/annotations/activity_net.v1-3.min.json'),
        blocked=str(DATA/'annotations/annotations/blocked.json'),
        out_dir=str(STAGING/'15fps_short256'), report_dir=str(WORK/'reports/data/anet_preparation'),
        temp_dir=str(STAGING/'tmp'), ffmpeg=str(ENV/'ffmpeg'), ffprobe=str(ENV/'ffprobe'),
        workers=workers, limit=None, raw_dir=[str(DATA/'raw_data/v1-3/train_val')],
        zip=[str(DATA/'downloads/hf_activitynet_snapshot/missing_files.zip')])
    blocked = set(json.loads(Path(args.blocked).read_text()))
    required = {k for k, v in json.loads(Path(args.annotation).read_text())['database'].items()
                if v['subset'] in ('training', 'validation') and k not in blocked}
    listing = json.loads(LISTING.read_text())
    groups = {}
    for subset in ('train', 'val'):
        entries = sorted((e for e in listing if e['path'].startswith(f'v1-2_{subset}.tar.gz.')),
                         key=lambda e: e['path'])
        for entry in entries:
            path = STAGING/'downloads'/entry['path']
            if not path.exists() or path.stat().st_size != entry['size']:
                raise RuntimeError(f'Archive is not completely downloaded: {path}')
        groups[subset] = [STAGING/'downloads'/e['path'] for e in entries]
    reports = Path(args.report_dir)
    reports.mkdir(parents=True, exist_ok=True)
    Path(args.temp_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    journal = reports/'prepared_videos.jsonl'
    done = set()
    if journal.exists():
        for line in journal.read_text().splitlines():
            row = json.loads(line)
            output = Path(args.out_dir)/f"v_{row['video_id']}.mp4"
            if row['status']=='PREPARED' and output.exists() and output.stat().st_size==row['bytes']:
                done.add(row['video_id'])
    seen = {subset: set() for subset in groups}
    failures = []
    with journal.open('a', encoding='utf-8') as receipts, ThreadPoolExecutor(args.workers) as executor:
        pending = {}

        def collect(completed):
            for future in completed:
                temporary, provenance = pending.pop(future)
                try:
                    row = future.result()
                    row['source'] = provenance
                    receipts.write(json.dumps(row)+'\n')
                    receipts.flush()
                    if row['status']=='PREPARED':
                        done.add(row['video_id'])
                    else:
                        failures.append(row)
                        print(json.dumps(row), flush=True)
                finally:
                    temporary.unlink(missing_ok=True)

        for subset, shards in groups.items():
            with closing(ShardReader(shards)) as reader, tarfile.open(fileobj=reader, mode='r|gz') as archive:
                for member in archive:
                    if not member.isfile() or not is_video(member.name):
                        continue
                    name = video_id(member.name)
                    seen[subset].add(name)
                    if name not in required or name in done:
                        continue
                    with tempfile.NamedTemporaryFile(suffix='.mp4', dir=args.temp_dir, delete=False) as temp:
                        temporary = Path(temp.name)
                        with archive.extractfile(member) as source:
                            shutil.copyfileobj(source, temp, 1024*1024)
                    future = executor.submit(convert, args, (name, dict(kind='file', path=str(temporary))))
                    pending[future] = (temporary, dict(kind='tar_shards', subset=subset,
                        archive_prefix=str(shards[0])[:-2], member=member.name))
                    if len(pending) >= 2*args.workers:
                        collect(wait(pending, return_when=FIRST_COMPLETED).done)
                    if len(seen[subset]) % 100 == 0:
                        print(json.dumps(dict(subset=subset, archive_videos_seen=len(seen[subset]),
                                              total_prepared=len(done))), flush=True)
            collect(wait(pending).done) if pending else None
    (reports/'archive_coverage.json').write_text(json.dumps(dict(
        archive_ids={k:sorted(v) for k,v in seen.items()}, failed_conversions=failures), indent=2)+'\n')
    result = finish_preparation(args)
    report = json.loads((reports/'preparation.json').read_text())
    # The paper handoff explicitly preserves all 43 downloaded archives.
    if report['status']=='READY' and ready_output:
        path=Path(ready_output);path.parent.mkdir(parents=True,exist_ok=True)
        value=dict(status='READY',prepared=report['prepared'],required=report['required'],
                   missing=report['missing'],archives_preserved=True,journal=str(journal),
                   preparation_report=str(reports/'preparation.json'),slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                   slurm_step_id=os.environ.get('SLURM_STEP_ID'))
        path.write_text(json.dumps(value,indent=2)+'\n')
    return result or report['status']!='READY'


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--ready-output',required=True)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--started-output')
    cli=parser.parse_args()
    if cli.dry_run:
        print(json.dumps(dict(archives=str(STAGING/'downloads'),listing=str(LISTING),
                             preserved=True,workers=cli.workers,ready_output=cli.ready_output,execute=False)))
    else:
        import fcntl
        if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Full data preparation requires its owned Slurm allocation')
        lock_path=STAGING/'paper_preparation.lock'
        with lock_path.open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            lock.write(str(os.getpid()));lock.flush()
            if cli.started_output:
                path=Path(cli.started_output);path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(dict(slurm_job_id=os.environ['SLURM_JOB_ID'],slurm_step_id=os.environ.get('SLURM_STEP_ID'),
                    workers=cli.workers,cpu_affinity=sorted(os.sched_getaffinity(0)),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    nice=os.getpriority(os.PRIO_PROCESS,0),started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),preparation_lock_acquired=True),indent=2)+'\n')
            raise SystemExit(main(cli.ready_output,cli.workers))
