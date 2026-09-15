"""Check the pending public payload for credentials and Git size blockers."""
import json
from pathlib import Path
import re
import subprocess
import zipfile
from snapshot_remote import SECRETS

BASE = Path(__file__).resolve().parent
REPO = BASE.parent.parent / "h65_clean_adatad/wtr_publication_20260915"
REFS = ["6e2fc7f78424e2485e8d33b079fac926eb0cf4a6", "27d557ee3bd38a52aec8caaf16860a95957b44b6", "df2b168ff109dda8838ef38bf5dbee048d14c70b"]
GENERIC = re.compile(rb"(?i)[\"']?(?:password|passwd|api_key|access_token|auth_token|secret_key)[\"']?\s*[:=]\s*[\"']([^\"'\r\n]{6,})[\"']")
issues = []

def inspect(data, label):
    if any(pattern.search(data) for pattern in SECRETS):
        issues.append({"path": label, "reason": "credential_format"})
    for match in GENERIC.finditer(data):
        value = match.group(1)
        if value.lower() not in {b"password", b"your_api_key", b"changeme", b"api_key"} and not any(marker in value for marker in (b"${", b"os.env", b"process.env", b"<", b"YOUR_", b"example", b"dummy", b"test_")):
            issues.append({"path": label, "reason": "credential_named_literal_review", "line": data[:match.start()].count(b"\n") + 1})

def main():
    objects = subprocess.check_output(["git", "rev-list", "--objects", *REFS, "--not", "--remotes=origin"], cwd=REPO).decode().splitlines()
    cat = subprocess.Popen(["git", "cat-file", "--batch"], cwd=REPO, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    blobs, total, largest = 0, 0, []
    for line in objects:
        parts = line.split(" ", 1)
        oid, label = parts[0], parts[-1]
        cat.stdin.write((oid + "\n").encode())
        cat.stdin.flush()
        header = cat.stdout.readline().decode().split()
        size = int(header[2])
        data = cat.stdout.read(size)
        cat.stdout.read(1)
        if header[1] != "blob":
            continue
        blobs += 1
        total += size
        largest.append({"path": label, "bytes": size})
        if size >= 100 * 1024 * 1024:
            issues.append({"path": label, "reason": "git_file_too_large", "bytes": size})
        inspect(data, label)
    cat.stdin.close()
    cat.wait()
    pending = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=REPO).decode().split("\0")
    for rel in filter(None, pending):
        path = REPO / rel
        data = path.read_bytes()
        inspect(data, rel)
        if len(data) >= 100 * 1024 * 1024:
            issues.append({"path": rel, "reason": "git_file_too_large", "bytes": len(data)})
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                for item in archive.infolist():
                    if not item.is_dir():
                        inspect(archive.read(item), rel + ":" + item.filename)
    result = {"new_history_blobs": blobs, "new_history_bytes": total, "largest_new_blobs": sorted(largest, key=lambda x: x["bytes"], reverse=True)[:8], "pending_files": len(list(filter(None, pending))), "issues": issues}
    (BASE / "source_payload_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if issues:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
