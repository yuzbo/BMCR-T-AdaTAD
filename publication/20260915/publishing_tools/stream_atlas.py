"""Verify and upload Atlas parts while the sequential SSH download continues."""
import concurrent.futures
import json
import shutil
import time
from prepare_records import BASE, PUB, prepare_archive, issues
from upload_records import assets, upload

def main():
    root = BASE / "remote/atlas"
    manifest_path = root / "atlas-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["credential_pattern_files"]:
        raise RuntimeError("Remote credential pattern findings require review")
    existing = assets()
    pending = list(manifest["archives"])
    active, results, uploaded = {}, [], []
    def handle(item):
        expected = {x["archive_path"]: x["bytes"] for x in manifest["files"] if x["archive"] == item["name"]}
        result = prepare_archive("atlas", item, expected)
        if issues:
            raise RuntimeError("Review credential-like literal findings before upload")
        return result, upload(root / item["name"], existing)
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        while pending or active:
            if time.monotonic() - started > 3600:
                raise RuntimeError("Download/upload did not finish within one hour; inspect partial receipts")
            for item in list(pending):
                path = root / item["name"]
                if len(active) < 3 and path.exists() and path.stat().st_size == item["bytes"]:
                    active[pool.submit(handle, item)] = item
                    pending.remove(item)
            for task in list(active):
                if task.done():
                    result, delivery = task.result()
                    results.append(result)
                    uploaded.append(delivery)
                    print(json.dumps({"verified_and_uploaded": result["archive"], "files": result["file_count"], "completed_parts": len(results), "total_parts": len(manifest["archives"])}), flush=True)
                    del active[task]
            if pending or active:
                time.sleep(2)
    if sum(x["file_count"] for x in results) != len(manifest["files"]):
        raise RuntimeError("Manifest file count mismatch")
    target = PUB / "manifests" / manifest_path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, target)
    receipt = {"server": "atlas", "status": "PASS", "archives": sorted(results, key=lambda x: x["archive"]), "file_count": len(manifest["files"]), "credential_like_findings": issues}
    (root / "archive_verification.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    (PUB / "manifests/atlas-verification.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    uploaded.append(upload(manifest_path, existing))
    (BASE / "upload_receipt_atlas.json").write_text(json.dumps(uploaded, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "atlas_files": len(manifest["files"]), "parts": len(results)}), flush=True)

if __name__ == "__main__":
    main()
