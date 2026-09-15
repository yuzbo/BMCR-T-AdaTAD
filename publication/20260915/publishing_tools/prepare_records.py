"""Verify archive membership and preserve small records for GitHub browsing."""
import argparse
import concurrent.futures
import gzip
import json
from pathlib import Path
import shutil
import tarfile
from audit_publication import inspect, issues

BASE = Path(__file__).resolve().parent
PUB = BASE.parent.parent / "h65_clean_adatad/wtr_publication_20260915/publication/20260915"

def prepare_archive(server, item, expected):
    path = BASE / "remote" / server / item["name"]
    if path.stat().st_size != item["bytes"]:
        raise RuntimeError(f"Transfer size mismatch: {path.name}")
    actual, browsable = {}, []
    with gzip.open(path, "rb") as stream:
        with tarfile.open(fileobj=stream, mode="r|") as archive:
            for member in archive:
                if not member.isfile():
                    raise RuntimeError(f"Unexpected non-file entry: {member.name}")
                actual[member.name] = member.size
                suffix = Path(member.name).suffix.lower()
                small_record = member.size <= 750000
                core_training_log = member.name.startswith("core/research/paper/runs/") and suffix in {".jsonl", ".log"} and member.size <= 8000000
                if (small_record or core_training_log) and "/windows/" not in member.name and suffix in {".json", ".jsonl", ".log", ".txt", ".md", ".sh", ".yaml", ".yml", ".csv"}:
                    data = archive.extractfile(member).read()
                    inspect(data, server + "/" + member.name)
                    target = PUB / "records" / server / member.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                    browsable.append(str(target.relative_to(PUB)))
        while stream.read(1024 * 1024):
            pass
    if actual != expected:
        raise RuntimeError(f"Archive membership/size mismatch: {path.name}")
    return {"archive": item["name"], "file_count": len(actual), "bytes": item["bytes"], "uncompressed_bytes": sum(actual.values()), "browsable_files": browsable, "gzip_crc_and_manifest_membership": "PASS"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("servers", nargs="+", choices=["4090", "a100", "atlas"])
    args = parser.parse_args()
    for server in args.servers:
        local = BASE / "remote" / server
        manifest_path = local / f"{server}-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["credential_pattern_files"]:
            raise RuntimeError("Remote credential pattern findings require review")
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            tasks = []
            for item in manifest["archives"]:
                expected = {row["archive_path"]: row["bytes"] for row in manifest["files"] if row["archive"] == item["name"]}
                tasks.append(pool.submit(prepare_archive, server, item, expected))
            for task in concurrent.futures.as_completed(tasks):
                result = task.result()
                results.append(result)
                print(json.dumps({k: v for k, v in result.items() if k != "browsable_files"}), flush=True)
        if issues:
            (local / "payload_review_findings.json").write_text(json.dumps(issues, indent=2), encoding="utf-8")
            raise RuntimeError(f"{server}: review literal credential-like matches before upload")
        if sum(x["file_count"] for x in results) != len(manifest["files"]):
            raise RuntimeError("Manifest file count mismatch")
        target = PUB / "manifests" / manifest_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, target)
        receipt = {"server": server, "status": "PASS", "archives": sorted(results, key=lambda x: x["archive"]), "file_count": len(manifest["files"]), "credential_like_findings": issues}
        (local / "archive_verification.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        (PUB / "manifests" / f"{server}-verification.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
