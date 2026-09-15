"""Collect Atlas's two root-level resource/order manifests as source metadata."""
import datetime as dt
import json
from pathlib import Path
import subprocess
from audit_publication import inspect, issues

BASE = Path(__file__).resolve().parent
PUB = BASE.parent.parent / "h65_clean_adatad/wtr_publication_20260915/publication/20260915"
root = PUB / "atlas/runtime"
root.mkdir(parents=True, exist_ok=True)
sources = []
for name in ("resources.json", "static_orders.json"):
    remote = "/root/autodl-tmp/wtr_characterization_20260915/" + name
    target = BASE / name
    command = ["scp", "-P", "44909", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3", "root@connect.nmb1.seetacloud.com:" + remote, str(target)]
    subprocess.run(command, check=True, timeout=120)
    data = target.read_bytes()
    json.loads(data)
    inspect(data, remote)
    if issues:
        raise RuntimeError("Credential-like content requires review")
    (root / name).write_bytes(data)
    sources.append({"source": remote, "published_path": "atlas/runtime/" + name, "bytes": len(data), "captured_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
(root / "provenance.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
print(json.dumps(sources, indent=2))
