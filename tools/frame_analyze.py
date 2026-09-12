"""Analyze real FPW records and emit publication-ready figures.

No values are imputed: records without the required fields simply do not
contribute points, and coverage.json records why a requested figure is absent.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THRESHOLDS = ["mAP@0.3", "mAP@0.4", "mAP@0.5", "mAP@0.6", "mAP@0.7"]

def _num(x):
    return x if isinstance(x, (int, float)) and x == x else None

def load_records(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path); obj = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and isinstance(obj.get("records"), list): return obj["records"]
    if isinstance(obj, dict) and isinstance(obj.get("results"), list):
        out=[]
        for r in obj["results"]:
            m=dict(r.get("metrics") or {})
            out.append({"id": r.get("id", f"{r.get('backbone','?')}_{r.get('method','?')}"),
                        "backbone": str(r.get("backbone", "")).lower(), "status":"complete",
                        "metrics":m, "gflops": r.get("gflops", r.get("matrix_conv_gflops")),
                        "latency_ms":r.get("latency_ms")})
        return out
    raise ValueError("manifest must contain records list (or legacy results list)")

def pareto(records):
    pts=[r for r in records if _num(r.get("gflops")) is not None and _num((r.get("metrics") or {}).get("average_mAP")) is not None]
    keep=[]
    for r in pts:
        x,y=r["gflops"],r["metrics"]["average_mAP"]
        if not any(q["gflops"] <= x and q["metrics"]["average_mAP"] >= y and (q["gflops"] < x or q["metrics"]["average_mAP"] > y) for q in pts): keep.append(r)
    return keep

def _save(fig, out, stem):
    for ext in ("pdf", "svg", "png"): fig.savefig(out/f"{stem}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)

def analyze(manifest, output, figspec=None, dry_run=False):
    out=Path(output); records=load_records(manifest)
    if not dry_run: out.mkdir(parents=True, exist_ok=True)
    coverage={"manifest":str(manifest),"record_count":len(records),"figures":{}}
    def status(name, n, reason): coverage["figures"][name]={"status":"available" if n else "missing","points":n,"reason":reason}
    valid=pareto(records); status("map_gflops_pareto",len(valid),"requires average_mAP and gflops")
    if valid and not dry_run:
        fig,ax=plt.subplots();
        for b in ("s","b"):
            rr=[r for r in valid if str(r.get("backbone","")).lower()==b]
            if rr: ax.scatter([r["gflops"] for r in rr],[r["metrics"]["average_mAP"] for r in rr],label=b.upper())
            for r in rr: ax.annotate(str(r.get("id","")),(r["gflops"],r["metrics"]["average_mAP"]))
        ax.set(xlabel="Compute (GFLOPs; 2× matrix/conv MAC)",ylabel="average mAP"); ax.legend(); _save(fig,out,"map_gflops_pareto")
    th=[r for r in records if all(_num((r.get("metrics") or {}).get(k)) is not None for k in THRESHOLDS)]
    status("five_threshold_map",len(th),"requires all five mAP thresholds")
    if th and not dry_run:
        fig,ax=plt.subplots();
        for r in th: ax.plot(THRESHOLDS,[r["metrics"][k] for k in THRESHOLDS],marker="o",label=str(r.get("id","")))
        ax.set(xlabel="IoU threshold",ylabel="mAP"); ax.legend(fontsize=7); _save(fig,out,"five_threshold_map")
    lat=[r for r in records if _num(r.get("latency_ms")) is not None]
    status("latency_cohort",len(lat),"latency is reported independently and never filters Pareto")
    if lat and not dry_run:
        fig,ax=plt.subplots(); ax.bar([str(r.get("id","")) for r in lat],[r["latency_ms"] for r in lat]); ax.set(xlabel="record",ylabel="latency (ms)"); plt.setp(ax.get_xticklabels(), rotation=30, ha="right"); _save(fig,out,"latency_cohort")
    curves=[]
    for r in records:
        c=r.get("training_curve") or r.get("curve")
        if isinstance(c,list) and c: curves.append((r,c))
    status("training_curve",len(curves),"requires record training_curve/curve list")
    if curves and not dry_run:
        fig,ax=plt.subplots();
        for r,c in curves: ax.plot([x.get("update",x.get("step",i)) for i,x in enumerate(c)],[x.get("average_mAP",x.get("mAP")) for x in c],label=str(r.get("id","")))
        ax.set(xlabel="update",ylabel="mAP"); ax.legend(fontsize=7); _save(fig,out,"training_curve")
    if not dry_run:
        (out/"source_manifest.json").write_text(json.dumps({"records":records},indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
        (out/"coverage.json").write_text(json.dumps(coverage,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return coverage

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",required=True); ap.add_argument("--figspec"); ap.add_argument("--output",required=True); ap.add_argument("--dry-run",action="store_true"); a=ap.parse_args()
    c=analyze(a.manifest,a.output,a.figspec,a.dry_run); print(json.dumps(c,ensure_ascii=False,indent=2))
if __name__ == "__main__": main()
