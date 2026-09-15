"""Capture experiment records without touching running jobs or copying weights."""
import argparse
from collections import defaultdict
import datetime as dt
import io
import json
import os
from pathlib import Path
import re
import tarfile

ROOTS = {
    "4090": {
        "core": "/data/run01/sczc063/yuzibo/wtr_fasttrack_20260915",
        "raw": "/data/run01/sczc063/yuzibo/wtr_raw_v1_20260915",
        "paper": "/data/run01/sczc063/yuzibo/h65_clean_adatad_20260910/paper_20260913",
        "support": "/data/run01/sczc063/yuzibo/h65_clean_adatad_20260910/support_review_20260914",
        "graph": "/data/run01/sczc063/yuzibo/h65_clean_adatad_20260910/graph_tad_20260914",
    },
    "a100": {
        "core": "/HOME/pxyai/pxyai_0057/HDD_POOL/yzb/wtr_fasttrack_20260915",
        "raw": "/HOME/pxyai/pxyai_0057/HDD_POOL/yzb/wtr_raw_v1_20260915",
    },
    "atlas": {"atlas": "/root/autodl-tmp/wtr_characterization_20260915"},
}
SUBDIRS = {
    "core": ["research/paper", "research/wtr_fasttrack", "logs", "runtime", "EVALUATION_REVISION", "WTR_FASTTRACK_SCIENCE_SHA", "source_revision.txt", "CODE_REVISION"],
    "raw": ["results", "logs", "runtime", "CODE_REVISION", "revision_27d557e/results", "revision_27d557e/logs", "revision_27d557e/runtime", "revision_27d557e/CODE_REVISION", "revision_27d557e/raw_course_4090.sh"],
    "paper": ["research/paper", "FASTTRACK_RETIREMENT.json", "source_revision.txt"],
    "support": ["research/paper", "source_revision.txt"],
    "graph": ["research/paper", "source_revision.txt"],
    "atlas": ["results", "analysis", "logs", "queue", "output", "CODE_REVISION", "PLOT_REVISION"],
}
SKIP_DIRS = {".git", ".ssh", "node_modules", "__pycache__", ".pytest_cache", ".venv", "venv", "envs", "assets", "checkpoints", "videos", "rawframes", "frames", "data"}
SKIP_SUFFIXES = {".pth", ".pt", ".ckpt", ".safetensors", ".bin", ".mp4", ".avi", ".mov", ".mkv", ".pyc", ".pyo", ".tar", ".gz", ".zip", ".7z", ".pem", ".key"}
SECRETS = [
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{40,}\b"),
]

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def gather(server):
    files, excluded, missing, seen = [], [], [], set()
    for component, root_text in ROOTS[server].items():
        root = Path(root_text)
        for sub in SUBDIRS[component]:
            start = root / sub
            if not start.exists():
                missing.append(str(start))
                continue
            candidates = []
            if start.is_file():
                candidates.append(start)
            else:
                for base, dirs, names in os.walk(start, followlinks=False):
                    dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not (Path(base) / d).is_symlink())
                    candidates.extend(Path(base) / name for name in sorted(names))
            for path in candidates:
                if str(path) in seen:
                    continue
                seen.add(str(path))
                rel = str(Path(component) / path.relative_to(root))
                if path.is_symlink() or not path.is_file():
                    excluded.append({"path": rel, "reason": "link_or_nonregular"})
                    continue
                if path.suffix.lower() in SKIP_SUFFIXES or path.name.startswith(".env") or path.name in {"id_rsa", "id_ed25519", "credentials", "credentials.json"}:
                    excluded.append({"path": rel, "reason": "weights_data_archives_cache_or_credentials"})
                    continue
                stat = path.stat()
                files.append({"path": str(path), "archive_path": rel, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return files, excluded, missing

def scan_secret(path):
    tail = b""
    with open(path, "rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                return False
            text = tail + block
            if any(rule.search(text) for rule in SECRETS):
                return True
            tail = text[-512:]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("server", choices=ROOTS)
    parser.add_argument("--output")
    args = parser.parse_args()
    started = now()
    files, excluded, missing = gather(args.server)
    summary = {"server": args.server, "capture_started_utc": started, "roots": ROOTS[args.server], "file_count": len(files), "uncompressed_bytes": sum(x["bytes"] for x in files), "largest_files": sorted(files, key=lambda x: x["bytes"], reverse=True)[:12], "missing_optional_paths": missing, "excluded_file_count": len(excluded)}
    totals = defaultdict(lambda: {"files": 0, "bytes": 0})
    for entry in files:
        key = "/".join(Path(entry["archive_path"]).parts[:3]) + " /*" + Path(entry["path"]).suffix
        totals[key]["files"] += 1
        totals[key]["bytes"] += entry["bytes"]
    summary["directory_totals"] = dict(sorted(totals.items(), key=lambda item: item[1]["bytes"], reverse=True))
    if not args.output:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit("Use a new empty output directory for an immutable snapshot")
    # Bounded parts keep uploads resumable. A single file is never split.
    batches, batch, size = [], [], 0
    for entry in files:
        if batch and size + entry["bytes"] > 400 * 1024 * 1024:
            batches.append(batch)
            batch, size = [], 0
        batch.append(entry)
        size += entry["bytes"]
    if batch:
        batches.append(batch)
    captured, blocked, changed, archives = [], [], [], []
    for index, group in enumerate(batches, 1):
        dest = out / f"{args.server}-experiment-records-part{index:02d}.tar.gz"
        with tarfile.open(dest, "w:gz", compresslevel=3) as archive:
            for entry in group:
                path = Path(entry["path"])
                if scan_secret(path):
                    blocked.append({"path": entry["archive_path"], "reason": "credential_pattern_requires_review"})
                    continue
                before = path.stat()
                info = archive.gettarinfo(str(path), arcname=entry["archive_path"])
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with path.open("rb") as source:
                    archive.addfile(info, source)
                after = path.stat()
                record = dict(entry, bytes=info.size, archive=dest.name, captured_utc=now())
                captured.append(record)
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    changed.append(entry["archive_path"])
        archives.append({"name": dest.name, "bytes": dest.stat().st_size, "file_count": sum(x["archive"] == dest.name for x in captured)})
        print(json.dumps({"archive_complete": archives[-1]}), flush=True)
    manifest = dict(summary, capture_finished_utc=now(), archives=archives, files=captured, excluded_files=excluded, credential_pattern_files=blocked, changed_while_captured=changed, snapshot_note="Running jobs were not paused. Each file is a point-in-time copy; partial last log records and unfinished stages may be present. Model weights, video/data assets, environment caches, credentials and redundant archives are excluded.")
    target = out / f"{args.server}-manifest.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(target), "captured_files": len(captured), "credential_pattern_files": blocked, "archives": archives}), flush=True)

if __name__ == "__main__":
    main()
