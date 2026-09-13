"""Relay the missing public ActivityNet v1.2 archives to N16R4 over SSH.

Run on the desktop, which has HTTPS access. N16R4 has no direct HTTPS route.
With Git Bash, set MSYS_NO_PATHCONV=1 to preserve the remote /data path.
Only train/validation shards from the recorded HF listing are downloaded.
Interrupted transfers retain their remote .part file and resume by byte offset.
"""
import argparse
import json
import shlex
import subprocess
import time
import urllib.request
from pathlib import Path


# N16R4 intermittently closes SSH for several consecutive attempts. Five rapid
# retries stopped both long transfers; keep their existing offsets and back off.
TRANSFER_ATTEMPTS = 60
RETRY_DELAY_SECONDS = 60


REMOTE_STAT = """
import json, pathlib, shutil, sys
p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True, exist_ok=True)
part=p.with_name(p.name+'.part')
print(json.dumps(dict(complete=p.stat().st_size if p.exists() else None,
    offset=part.stat().st_size if part.exists() else 0,
    free=shutil.disk_usage(p.parent).free)))
"""

REMOTE_RECEIVE = """
import pathlib, sys
p=pathlib.Path(sys.argv[1]); expected=int(sys.argv[2]); offset=int(sys.argv[3])
part=p.with_name(p.name+'.part')
actual=part.stat().st_size if part.exists() else 0
if actual != offset: raise RuntimeError('Partial file changed before transfer')
with part.open('ab') as f:
    while True:
        chunk=sys.stdin.buffer.read(4*1024*1024)
        if not chunk: break
        f.write(chunk)
size=part.stat().st_size
if size > expected: raise RuntimeError('Transfer exceeds published file size')
if size == expected: part.rename(p)
print(size)
"""


def ssh_command(args, program, *values):
    remote = shlex.join(['python3', '-c', program, *map(str, values)])
    return [args.ssh, '-F', args.ssh_config, '-o', 'BatchMode=yes',
            '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=30', args.host, remote]


def download(args, entry):
    name, expected = entry['path'], entry['size']
    destination = args.remote_dir.rstrip('/') + '/' + name
    for attempt in range(1, TRANSFER_ATTEMPTS + 1):
        # Remote stat is part of the transfer, and can fail before HTTP begins.
        # Previously a transient SSH timeout here bypassed all resume retries.
        try:
            info = json.loads(subprocess.check_output(
                ssh_command(args, REMOTE_STAT, destination), text=True,
                stderr=subprocess.PIPE,timeout=60))
        except (OSError,ValueError,subprocess.SubprocessError) as error:
            print(json.dumps(dict(file=name,attempt=attempt,phase='remote_stat',error=str(error))),flush=True)
            if attempt==TRANSFER_ATTEMPTS:raise
            time.sleep(RETRY_DELAY_SECONDS)
            continue
        if info['complete'] is not None:
            if info['complete'] != expected:
                raise RuntimeError(f'{name}: existing file size differs from HF listing')
            return dict(file=name, bytes=expected, status='PRESENT')
        offset = info['offset']
        if offset == expected:
            subprocess.run(ssh_command(args, REMOTE_RECEIVE, destination, expected, offset),
                           stdin=subprocess.DEVNULL, check=True, capture_output=True)
            return dict(file=name, bytes=expected, status='RESUMED_COMPLETE')
        if offset > expected:
            raise RuntimeError(f'{name}: partial file exceeds the published size')
        if info['free'] - (expected - offset) < args.reserve_gib * 1024**3:
            raise RuntimeError('Not enough free space after the training-space reserve')
        url = 'https://huggingface.co/datasets/YimuWang/ActivityNet/resolve/main/' + name
        # The offset also distinguishes cached redirects for resumed Range requests.
        url += f'?download=true&offset={offset}&request_time={time.time_ns()}'
        receiver = None
        start = time.monotonic()
        try:
            request = urllib.request.Request(url, headers={
                'Range': f'bytes={offset}-', 'User-Agent': 'ActivityNet-data-preparation'})
            with urllib.request.urlopen(request, timeout=60) as response:
                content_range = response.headers.get('Content-Range', '')
                if response.status != 206 or not content_range.startswith(f'bytes {offset}-'):
                    raise RuntimeError(f'Unexpected range response: {response.status} {content_range}')
                if int(content_range.rsplit('/', 1)[-1]) != expected:
                    raise RuntimeError('HTTP size differs from recorded HF listing')
                receiver = subprocess.Popen(ssh_command(args, REMOTE_RECEIVE,
                    destination, expected, offset), stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                transferred = offset
                last_report = start
                while True:
                    chunk = response.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    receiver.stdin.write(chunk)
                    transferred += len(chunk)
                    if time.monotonic() - last_report >= 30:
                        print(json.dumps(dict(file=name, bytes_sent=transferred,
                            expected=expected, elapsed_seconds=round(time.monotonic()-start, 1))), flush=True)
                        last_report = time.monotonic()
                receiver.stdin.close()
                receiver.stdin = None
                output, error = receiver.communicate(timeout=60)
                if receiver.returncode:
                    raise RuntimeError(error.decode(errors='replace'))
                if int(output.strip()) == expected:
                    return dict(file=name, bytes=expected, status='DOWNLOADED')
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            print(json.dumps(dict(file=name, attempt=attempt, error=str(error))), flush=True)
            # An observed broken pipe otherwise left close()/flush waiting on
            # the stalled SSH child. Stop it before cleaning up buffered stdin.
            if receiver is not None and receiver.poll() is None:
                receiver.terminate()
        finally:
            if receiver is not None and receiver.poll() is None:
                if receiver.stdin is not None:
                    try:
                        receiver.stdin.close()
                    except (BrokenPipeError,OSError):
                        pass
                    receiver.stdin = None
                try:
                    receiver.communicate(timeout=60)
                except subprocess.TimeoutExpired:
                    receiver.terminate()
                    receiver.communicate()
        if attempt < TRANSFER_ATTEMPTS:
            time.sleep(RETRY_DELAY_SECONDS)
    raise RuntimeError(f'{name}: transfer remains incomplete after {TRANSFER_ATTEMPTS} attempts')


def main(args):
    if not args.remote_dir.startswith('/'):
        raise ValueError('Use an absolute remote POSIX path; set MSYS_NO_PATHCONV=1 in Git Bash')
    entries = json.loads(Path(args.listing).read_text())
    entries = [e for e in entries if e['type'] == 'file' and
               any(e['path'].startswith(f'v1-2_{subset}.tar.gz.') for subset in args.subsets)]
    entries.sort(key=lambda e: e['path'])
    if args.limit:
        entries = entries[:args.limit]
    receipt = Path(args.receipt)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        result = download(args, entry)
        print(json.dumps(result), flush=True)
        with receipt.open('a', encoding='utf-8') as f:
            f.write(json.dumps(result) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--listing', required=True)
    parser.add_argument('--ssh-config', required=True)
    parser.add_argument('--ssh', default='ssh')
    parser.add_argument('--host', default='bcr-4090')
    parser.add_argument('--remote-dir', required=True)
    parser.add_argument('--subsets', nargs='+', choices=['train', 'val'], default=['train', 'val'])
    parser.add_argument('--reserve-gib', type=int, default=150)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--receipt', required=True)
    main(parser.parse_args())
