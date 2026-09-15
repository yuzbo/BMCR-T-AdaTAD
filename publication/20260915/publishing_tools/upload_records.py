"""Upload only verified snapshot parts; inspect existing asset state on retries."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import shutil
import subprocess
import time

BASE = Path(__file__).resolve().parent
REPO = "yuzbo/BMCR-T-AdaTAD"
TAG = "wtr-snapshot-20260915"
GH = shutil.which("gh")

def assets():
    data = subprocess.check_output([GH, "release", "view", TAG, "--repo", REPO, "--json", "assets"], text=True)
    return {x["name"]: x for x in json.loads(data)["assets"]}

def upload(path, existing):
    prior = existing.get(path.name)
    if prior and prior["state"] == "uploaded" and prior["size"] == path.stat().st_size:
        return {"name": path.name, "bytes": path.stat().st_size, "status": "already_uploaded"}
    for attempt in range(3):
        command = [GH, "release", "upload", TAG, str(path), "--repo", REPO]
        if prior:
            command.append("--clobber")
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=600)
        if result.returncode == 0:
            return {"name": path.name, "bytes": path.stat().st_size, "status": "uploaded"}
        prior = assets().get(path.name)
        if prior and prior["state"] == "uploaded" and prior["size"] == path.stat().st_size:
            return {"name": path.name, "bytes": path.stat().st_size, "status": "uploaded_after_uncertain_response"}
        print(f"Retrying {path.name} after upload exit {result.returncode}", flush=True)
        time.sleep(2)
    raise RuntimeError(f"Upload failed: {path.name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("servers", nargs="+", choices=["4090", "a100", "atlas"])
    args = parser.parse_args()
    pending = []
    for server in args.servers:
        root = BASE / "remote" / server
        receipt = json.loads((root / "archive_verification.json").read_text())
        if receipt["status"] != "PASS":
            raise RuntimeError("Archives must be verified first")
        pending += [root / f"{server}-manifest.json", *[root / x["archive"] for x in receipt["archives"]]]
    existing = assets()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        tasks = [pool.submit(upload, path, existing) for path in pending]
        for task in concurrent.futures.as_completed(tasks):
            result = task.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    (BASE / ("upload_receipt_" + "_".join(args.servers) + ".json")).write_text(json.dumps(results, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
