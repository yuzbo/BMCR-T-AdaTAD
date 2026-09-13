"""Prepare available full ActivityNet videos at 15 fps and short side 256.

This is data preparation, never a subset performance experiment. The receipt
lists every missing official ID; incomplete coverage cannot become READY.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import zipfile


def video_id(path):
    name = Path(path).stem
    return name[2:] if name.startswith('v_') else name


def probe(executable, path):
    output = subprocess.check_output([executable, '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,avg_frame_rate,nb_frames,duration',
        '-of', 'json', str(path)], stderr=subprocess.STDOUT, timeout=60)
    stream = json.loads(output)['streams'][0]
    return dict(width=int(stream['width']), height=int(stream['height']),
        fps=float(Fraction(stream['avg_frame_rate'])),
        frames=int(stream['nb_frames']) if stream.get('nb_frames', 'N/A') != 'N/A' else None,
        duration=float(stream['duration']) if stream.get('duration', 'N/A') != 'N/A' else None)


def convert(args, item):
    name, source = item
    output = Path(args.out_dir) / f'v_{name}.mp4'
    partial = output.with_suffix('.part.mp4')
    begin = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f'{name}_', dir=args.temp_dir) as temporary:
        try:
            if source['kind'] == 'zip':
                input_path = Path(temporary) / f'v_{name}.mp4'
                with zipfile.ZipFile(source['path']) as archive:
                    with archive.open(source['member']) as incoming, input_path.open('wb') as destination:
                        shutil.copyfileobj(incoming, destination, 1024 * 1024)
            else:
                input_path = Path(source['path'])
            before = probe(args.ffprobe, input_path)
            command = [args.ffmpeg, '-nostdin', '-v', 'error', '-xerror', '-y',
                '-threads', '2', '-i', str(input_path), '-map', '0:v:0',
                '-vf', "fps=15,scale=w='if(gte(iw,ih),-2,256)':h='if(gte(iw,ih),256,-2)'",
                '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-pix_fmt', 'yuv420p',
                '-an', '-movflags', '+faststart', '-threads', '2', str(partial)]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=900)
            if completed.returncode:
                raise RuntimeError(completed.stderr[-4000:])
            after = probe(args.ffprobe, partial)
            if abs(after['fps'] - 15) > 1e-6 or min(after['width'], after['height']) != 256:
                raise RuntimeError(f'Preprocessing dimensions/rate do not match: {after}')
            if not after['frames'] or not after['duration']:
                raise RuntimeError('No complete video stream in converted output')
            partial.replace(output)
            return dict(video_id=name, status='PREPARED', source=source, source_video=before,
                prepared_video=after, bytes=output.stat().st_size,
                seconds=round(time.monotonic() - begin, 3))
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            partial.unlink(missing_ok=True)
            return dict(video_id=name, status='FAILED', source=source, error=str(error))


def write_coverage_report(args, prepared, failures):
    database = json.loads(Path(args.annotation).read_text())['database']
    blocked = set(json.loads(Path(args.blocked).read_text()))
    required = {k: v for k, v in database.items()
                if v['subset'] in ('training', 'validation') and k not in blocked}
    missing = {subset: sorted(k for k, v in required.items()
                             if v['subset'] == subset and k not in prepared)
               for subset in ('training', 'validation')}
    report = dict(status='READY' if not any(missing.values()) else 'INCOMPLETE',
        required={subset: sum(v['subset'] == subset for v in required.values()) for subset in missing},
        prepared={subset: sum(v['subset'] == subset and k in prepared for k, v in required.items())
                  for subset in missing},
        missing=missing, failures=failures, output_directory=str(args.out_dir),
        preprocessing=dict(fps=15, short_side=256, codec='libx264', crf=18, preset='fast',
            pixel_format='yuv420p', audio=False,
            note='Matches published fps/size; official unpublished encoding settings are unknown.'),
        journal=str(Path(args.report_dir)/'prepared_videos.jsonl'), source_annotation=args.annotation,
        blocked=args.blocked, is_performance_experiment=False)
    (Path(args.report_dir)/'preparation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('missing','failures')}),flush=True)
    return report


def main(args):
    database = json.loads(Path(args.annotation).read_text())['database']
    blocked = set(json.loads(Path(args.blocked).read_text()))
    required = {k: v for k, v in database.items()
                if v['subset'] in ('training', 'validation') and k not in blocked}
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    reports = Path(args.report_dir)
    reports.mkdir(parents=True, exist_ok=True)
    Path(args.temp_dir).mkdir(parents=True, exist_ok=True)
    journal = reports / 'prepared_videos.jsonl'
    prepared = {}
    if journal.exists():
        for line in journal.read_text().splitlines():
            row = json.loads(line)
            target = output / f"v_{row['video_id']}.mp4"
            if row['status'] == 'PREPARED' and target.exists() and target.stat().st_size == row['bytes']:
                prepared[row['video_id']] = row
    sources = {}
    for directory in args.raw_dir:
        for source in Path(directory).rglob('*.mp4'):
            name = video_id(source)
            if name in required:
                sources.setdefault(name, dict(kind='file', path=str(source)))
    for archive_path in args.zip:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.namelist():
                name = video_id(member)
                if member.endswith('.mp4') and name in required:
                    sources.setdefault(name, dict(kind='zip', path=archive_path, member=member))
    # Recover missing videos from the local supplement before re-encoding existing raw files.
    items = sorted(((k, v) for k, v in sources.items() if k not in prepared),
                   key=lambda pair: (pair[1]['kind'] != 'zip', pair[0]))
    if args.limit:
        items = items[:args.limit]
    failures = []
    print(json.dumps(dict(required=len(required), available_sources=len(sources),
                         previously_prepared=len(prepared), scheduled=len(items))), flush=True)
    with journal.open('a', encoding='utf-8') as receipts, ThreadPoolExecutor(args.workers) as executor:
        for index, row in enumerate(executor.map(lambda item: convert(args, item), items), 1):
            receipts.write(json.dumps(row) + '\n')
            receipts.flush()
            if row['status'] == 'PREPARED':
                prepared[row['video_id']] = row
            else:
                failures.append(row)
                print(json.dumps(row), flush=True)
            if index % 25 == 0 or index == len(items):
                print(json.dumps(dict(processed=index, scheduled=len(items),
                    prepared=len(prepared), failed=len(failures))), flush=True)
    write_coverage_report(args,prepared,failures)
    return bool(failures)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', required=True)
    parser.add_argument('--blocked', required=True)
    parser.add_argument('--raw-dir', action='append', default=[])
    parser.add_argument('--zip', action='append', default=[])
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--report-dir', required=True)
    parser.add_argument('--temp-dir', required=True)
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--ffprobe', default='ffprobe')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, help='Data-preparation smoke check only; never marks incomplete data READY')
    raise SystemExit(main(parser.parse_args()))
