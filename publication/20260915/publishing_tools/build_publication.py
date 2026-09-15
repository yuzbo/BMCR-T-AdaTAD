"""Collect publication copies, preserving exact independent scientific revisions."""
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

BASE = Path(__file__).resolve().parent
WORKSPACE = BASE.parent.parent
TREES = WORKSPACE / "h65_clean_adatad"
DEST = TREES / "wtr_publication_20260915"
PUB = DEST / "publication" / "20260915"
REPO = "https://github.com/yuzbo/BMCR-T-AdaTAD"
SOURCES = {
    "core": {"directory": "wtr_fasttrack_20260915", "branch": "codex/wtr-fasttrack", "commit": "6e2fc7f78424e2485e8d33b079fac926eb0cf4a6", "science_commit": "40552945ad5d56b7404f833cd86f998e8558e8b1", "note": "6e2fc7f changes evaluation logging only; scientific model/training provenance remains 4055294."},
    "raw": {"directory": "wtr_raw_v1_20260915", "branch": "codex/wtr-raw-v1", "commit": "27d557ee3bd38a52aec8caaf16860a95957b44b6", "reader_initial_commit": "1259d25bd49291aaf6ef0aceb0bc07ecae0aa222"},
    "atlas": {"directory": "wtr_characterization_20260915", "branch": "codex/wtr-characterization-20260915", "commit": "df2b168ff109dda8838ef38bf5dbee048d14c70b", "measurement_commit": "a50b84db1bcdafd86ceec4d9737156807f9daa9a", "temporal_plot_commit": "6dbd246", "cost_ledger_commit": "c91cbd7"},
}
copied = []
rewritten = []

def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    copied.append({"source": str(source), "published_path": target.relative_to(DEST).as_posix(), "bytes": target.stat().st_size})

def copy_tree(source, target):
    for base, dirs, names in os.walk(source, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in {"node_modules", "__pycache__", ".git", ".pytest_cache"} and not (Path(base) / d).is_symlink())
        for name in sorted(names):
            src = Path(base) / name
            if src.is_symlink() or src.suffix in {".pyc", ".pyo", ".pth", ".pt", ".ckpt"}:
                continue
            copy_file(src, target / src.relative_to(source))

def rewrite_links(path):
    original = path.read_text(encoding="utf-8-sig")
    def target(match):
        text = match.group(1).replace("\\", "/")
        fragment = ""
        hit = re.search(r":(\d+)$", text)
        if hit:
            fragment = "#L" + hit.group(1)
            text = text[:hit.start()]
        reports_prefix = WORKSPACE.as_posix() + "/reports/"
        if text.startswith(reports_prefix):
            candidate = PUB / "reports" / text[len(reports_prefix):]
            if candidate.exists():
                return "](" + Path(os.path.relpath(candidate, path.parent)).as_posix() + fragment + ")"
        for source in SOURCES.values():
            prefix = (TREES / source["directory"]).as_posix() + "/"
            if text.startswith(prefix):
                suffix = text[len(prefix):]
                if source is SOURCES["core"] and (DEST / suffix).exists():
                    return "](" + Path(os.path.relpath(DEST / suffix, path.parent)).as_posix() + fragment + ")"
                return "](" + REPO + "/blob/" + source["commit"] + "/" + suffix + fragment + ")"
        return match.group(0)
    updated = re.sub(r"\]\(<?((?:C:|c:)[^\n)>]+)>?\)", target, original)
    if updated != original:
        path.write_text(updated, encoding="utf-8")
        rewritten.append(path.relative_to(DEST).as_posix())

def main():
    PUB.mkdir(parents=True, exist_ok=True)
    for source in SOURCES.values():
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=TREES / source["directory"], text=True).strip()
        if actual != source["commit"]:
            raise RuntimeError(f"Source advanced: {source['directory']}: inspect before capture ({actual})")
    for source in sorted((WORKSPACE / "reports").glob("wtr_*")):
        if source.is_dir():
            copy_tree(source, PUB / "reports" / source.name)
    core = TREES / SOURCES["core"]["directory"]
    for name in ("EVALUATION_REVISION", "WTR_FASTTRACK_SCIENCE_SHA"):
        copy_file(core / name, DEST / name)
    copy_tree(core / "research/wtr_fasttrack/reviews", DEST / "research/wtr_fasttrack/reviews")
    atlas = TREES / SOURCES["atlas"]["directory"]
    tracked = subprocess.check_output(["git", "ls-files", "-z", "research/atlas_20260915"], cwd=atlas).decode().split("\0")
    for rel in filter(None, tracked):
        copy_file(atlas / rel, PUB / "atlas" / rel)
    for name in ("CODE_REVISION", "PLOT_REVISION"):
        copy_file(atlas / "research/atlas_20260915" / name, PUB / "atlas" / name)
    copy_tree(atlas / "output", PUB / "figures")
    copy_file(TREES / SOURCES["raw"]["directory"] / "CODE_REVISION", PUB / "raw/CODE_REVISION")
    for name in ("snapshot_remote.py", "collect_remote.py", "build_publication.py"):
        copy_file(BASE / name, PUB / "publishing_tools" / name)
    copy_file(DEST / "README.md", PUB / "README_before_publication.md")
    for path in PUB.rglob("*.md"):
        rewrite_links(path)
    manifest = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "repository": REPO, "publication_branch": "codex/wtr-publication-20260915", "sources": SOURCES, "local_files": copied, "markdown_links_rewritten_in_publication_copies": rewritten, "notes": ["Source branches are preserved independently; the publication root uses Core 6e2fc7f. Switch to the exact Raw or Atlas branch to run those lines.", "No model weights, training videos or local dependency trees are included.", "Experiment outputs are temporal snapshots, not claims that unfinished scientific stages have passed."]}
    (PUB / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"copied_files": len(copied), "copied_bytes": sum(x["bytes"] for x in copied), "rewritten_markdown_files": len(rewritten)}, indent=2))

if __name__ == "__main__":
    main()
