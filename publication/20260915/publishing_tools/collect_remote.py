"""Inventory or download a publication snapshot through existing SSH access."""
import argparse
import concurrent.futures
import datetime as dt
import json
from pathlib import Path
import shutil
import subprocess
import time

BASE = Path(__file__).resolve().parent
CONFIG = "C:/Users/skywalker/Documents/ChatGPT/refine-tad/V3/motivation/ssh_config"
SERVERS = {
    "4090": {"host": "bcr-4090", "options": ["-F", CONFIG], "python": "/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python"},
    "a100": {"host": "bcr-a100", "options": ["-F", CONFIG], "python": "/HOME/pxyai/pxyai_0057/HDD_POOL/yzb/geosparse_tad_20260907/envs/opentad/bin/python"},
    "atlas": {"host": "root@connect.nmb1.seetacloud.com", "options": ["-p", "44909"], "python": "/root/autodl-tmp/envs/opentad/bin/python"},
}

def run_remote(name, capture):
    cfg = SERVERS[name]
    dest = BASE / "remote" / name
    dest.mkdir(parents=True, exist_ok=True)
    script = (BASE / "snapshot_remote.py").read_bytes()
    ssh = [shutil.which("ssh"), *cfg["options"], "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", cfg["host"]]
    for attempt in range(1, 4):
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        remote_dir = f"/tmp/h65-github-{stamp}-{name}-{attempt}"
        command = f"{cfg['python']} - {name}" + (f" --output {remote_dir}" if capture else "")
        proc = subprocess.run([*ssh, command], input=script, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (dest / f"capture-attempt{attempt}.txt").write_bytes(proc.stdout + proc.stderr)
        if proc.returncode == 0:
            break
        print(f"{name}: SSH/capture attempt {attempt} failed (exit {proc.returncode}); retrying read/snapshot only", flush=True)
        time.sleep(2)
    else:
        raise RuntimeError(f"{name}: capture failed; see local attempt logs")
    if not capture:
        data = json.loads(proc.stdout)
        (dest / "inventory.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(json.dumps({"server": name, "file_count": data["file_count"], "uncompressed_bytes": data["uncompressed_bytes"], "largest_files": [{"path": x["archive_path"], "bytes": x["bytes"]} for x in data["largest_files"][:5]], "missing_optional_paths": data["missing_optional_paths"]}), flush=True)
        return
    receipt = json.loads(proc.stdout.splitlines()[-1])
    if receipt["credential_pattern_files"]:
        raise RuntimeError(f"{name}: credential-like files excluded; inspect capture receipt before publishing")
    scp_options = ["-P" if x == "-p" else x for x in cfg["options"]]
    for filename in [f"{name}-manifest.json", *[x["name"] for x in receipt["archives"]]]:
        for attempt in range(3):
            cp = subprocess.run([shutil.which("scp"), *scp_options, "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", f"{cfg['host']}:{remote_dir}/{filename}", str(dest / filename)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if cp.returncode == 0:
                break
            time.sleep(2)
        else:
            raise RuntimeError(f"{name}: download failed: {filename}")
        print(json.dumps({"downloaded": str(dest / filename), "bytes": (dest / filename).stat().st_size}), flush=True)
    (dest / "capture-location.json").write_text(json.dumps({"server": name, "remote_directory": remote_dir, "receipt": receipt}, indent=2), encoding="utf-8")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--servers", nargs="+", default=list(SERVERS), choices=SERVERS)
    args = parser.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.servers)) as pool:
        tasks = {pool.submit(run_remote, name, args.capture): name for name in args.servers}
        for future in concurrent.futures.as_completed(tasks):
            future.result()

if __name__ == "__main__":
    main()
