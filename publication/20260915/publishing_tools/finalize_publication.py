"""Write the final publication index after GitHub confirms every asset."""
import datetime as dt
import json
from pathlib import Path
import shutil
import subprocess

BASE = Path(__file__).resolve().parent
REPO = BASE.parent.parent / "h65_clean_adatad/wtr_publication_20260915"
PUB = REPO / "publication/20260915"
URL = "https://github.com/yuzbo/BMCR-T-AdaTAD"
TAG = "wtr-snapshot-20260915"
DOWNLOAD = URL + "/releases/download/" + TAG + "/"

def clock(text):
    return dt.datetime.fromisoformat(text).astimezone(dt.timezone(dt.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")

def main():
    manifests, expected = {}, {}
    for server in ("4090", "a100", "atlas"):
        root = BASE / "remote" / server
        data = json.loads((root / f"{server}-manifest.json").read_text())
        check = json.loads((root / "archive_verification.json").read_text())
        if check["status"] != "PASS":
            raise RuntimeError(f"{server}: verification incomplete")
        manifests[server] = data
        expected[f"{server}-manifest.json"] = (root / f"{server}-manifest.json").stat().st_size
        expected.update({x["name"]: x["bytes"] for x in data["archives"]})
    bundle = BASE / "wtr-core-raw-atlas-source.bundle"
    expected[bundle.name] = bundle.stat().st_size
    release = json.loads(subprocess.check_output(["gh", "release", "view", TAG, "--repo", "yuzbo/BMCR-T-AdaTAD", "--json", "assets,tagName,isDraft,url"], text=True))
    actual = {item["name"]: item for item in release["assets"]}
    problems = [name for name, size in expected.items() if name not in actual or actual[name]["size"] != size or actual[name]["state"] != "uploaded"]
    if problems:
        raise RuntimeError("Incomplete GitHub assets: " + ", ".join(problems))
    browsable = sorted(path for path in (PUB / "records").rglob("*") if path.is_file())
    total_files = sum(len(data["files"]) for data in manifests.values())
    raw_bytes = sum(row["bytes"] for data in manifests.values() for row in data["files"])
    compressed_bytes = sum(row["bytes"] for data in manifests.values() for row in data["archives"])
    lines = ["# 实验记录与完整源码下载", "", f"本次快照包含 **{total_files:,} 个原始实验文件**，解压后约 **{raw_bytes / 1e9:.2f} GB**；日志与测量记录压缩后约 **{compressed_bytes / 1e9:.2f} GB**，共 70 个独立 tar.gz 分卷。另附三条源码分支及完整提交历史的 Git bundle（{bundle.stat().st_size / 1e6:.1f} MB）。大小使用十进制单位。", "", "[Release 全部附件](" + URL + "/releases/tag/" + TAG + ") · [可浏览的小型日志、配置和回执](records) · [逐文件清单](manifests)", "", "## 采集范围与时间", "", "下表时间为北京时间（UTC+8）。开始时枚举快照文件范围，随后逐文件复制；实验始终继续运行。每个 manifest 的 `captured_utc` 是逐文件复制时间，`mtime_ns` 是初始枚举元数据，`bytes` 是实际归档大小。采集开始后新增的文件不属于这次快照；较晚复制的现有日志可能包含后续追加内容。", "", "| 服务器 | 文件数 | 原始 GB | 压缩 MB | 分卷数 | 采集开始 | 采集结束 | 清单 |", "|---|---:|---:|---:|---:|---|---|---|"]
    for server, data in manifests.items():
        lines.append(f"| {server} | {len(data['files']):,} | {sum(x['bytes'] for x in data['files']) / 1e9:.3f} | {sum(x['bytes'] for x in data['archives']) / 1e6:.2f} | {len(data['archives'])} | {clock(data['capture_started_utc'])} | {clock(data['capture_finished_utc'])} | [{server} manifest](manifests/{server}-manifest.json) |")
    lines += ["", "4090 范围包含新 Core、Raw 初版与 27d557e 修正版，以及 paper、support、Graph 旧课程的训练、评测、失败、取消和部署记录。A100 保留课程排队、配置、部署与迁移回执。Atlas 包含 `results/` 逐窗口记录、`analysis/`、`logs/`、`queue/` 和 `output/`。完整路径和排除项见 manifest。", "", "模型权重、checkpoint、视频与帧数据、依赖环境、密钥、缓存字节码、已有重复打包文件未上传。源码单独通过 Git 分支和 bundle 发布。所有已选记录保持原始内容；独立 ZIP 设计原件也保存在报告目录。", "", "## 常用浏览入口", "", "- [Core D-U 完整训练日志](records/4090/core/research/paper/runs/wtr_d_u_s42/train.jsonl)", "- [Core D-V 完整训练日志](records/4090/core/research/paper/runs/wtr_d_v_s42/train.jsonl)", "- [Core 日志、评测和预检回执](records/4090/core)", "- [Raw-v1 验证、mini-bank 和诊断](records/4090/raw)", "- [A100 部署与迁移](records/a100)", "- [Atlas 可浏览记录](records/atlas)", "", f"Git 中额外提供 {len(browsable):,} 份可浏览记录：通常是小于 750 KB 的日志、JSON、脚本与文本，并包含两条当前 Core 完整训练日志。大型逐窗口结果和预测输出保存在 Release 分卷中。", "", "## 一次下载完整源码", "", "[wtr-core-raw-atlas-source.bundle](" + DOWNLOAD + bundle.name + ") 保留 Core、Raw、Atlas 三条独立分支及其 Git 历史。解包不需要连接 GitHub：", "", "```bash", "git clone --branch codex/wtr-fasttrack wtr-core-raw-atlas-source.bundle wtr-source", "cd wtr-source", "git switch codex/wtr-raw-v1", "# git switch codex/wtr-characterization-20260915", "```", "", "该 bundle 对应 Core `6e2fc7f`、Raw `27d557e`、Atlas `df2b168`。发布报告和图也可从本 Release 自动提供的 Source code ZIP 下载。", "", "## 下载与解压原始记录", "", "各分卷都是独立的 tar.gz 文件，不能拼接。可只下载所需服务器；同一服务器的分卷解压到同一目录。以下示例恢复 4090 记录：", "", "```bash", "gh release download " + TAG + " --repo yuzbo/BMCR-T-AdaTAD --pattern '4090-*' --dir records-4090", "mkdir -p snapshot-4090", "for part in records-4090/*.tar.gz; do", '  tar -xzf "$part" -C snapshot-4090', "done", "```", "", "Atlas 用 `--pattern 'atlas-*'` 并解压到独立目录；A100 同理。Atlas 测量文件恢复后可用首页的命令重建图。分卷大小按 [GitHub Release 文件规则](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)控制。", "", "| 文件 | 压缩 MB | 文件数 |", "|---|---:|---:|"]
    for server, data in manifests.items():
        for item in data["archives"]:
            lines.append(f"| [{item['name']}]({DOWNLOAD}{item['name']}) | {item['bytes'] / 1e6:.2f} | {item['file_count']} |")
    lines += ["", "每卷已核对 gzip 完整性、逐文件路径与大小和 manifest 一致性；GitHub 返回的附件名称、大小和 uploaded 状态也已核对。科学结论沿用相应实验报告的边界，发布核验不替代科学验证。", ""]
    (PUB / "RECORDS.md").write_text("\n".join(lines), encoding="utf-8")
    manifest_path = PUB / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest["local_files"]:
        row.setdefault("source_bytes", row["bytes"])
        row["bytes"] = (REPO / row["published_path"]).stat().st_size
    manifest["publication_branches"] = ["codex/publication", "codex/wtr-publication-20260915"]
    manifest["remote_records"] = {"files": total_files, "uncompressed_bytes": raw_bytes, "compressed_bytes": compressed_bytes, "archive_parts": 70, "browsable_files": len(browsable), "server_manifests": [f"manifests/{server}-manifest.json" for server in manifests], "source_bundle": {"name": bundle.name, "bytes": bundle.stat().st_size, "url": DOWNLOAD + bundle.name}, "release_url": URL + "/releases/tag/" + TAG}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    receipt = {"checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "status": "all_expected_assets_uploaded", "release_tag": TAG, "assets": [{"name": name, "bytes": size, "state": actual[name]["state"]} for name, size in expected.items()], "total_assets": len(expected), "total_bytes": sum(expected.values())}
    (PUB / "manifests/github_asset_verification.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    for name in ("snapshot_remote.py", "collect_remote.py", "build_publication.py", "prepare_records.py", "upload_records.py", "stream_atlas.py", "audit_publication.py", "finalize_publication.py", "collect_atlas_runtime.py"):
        shutil.copy2(BASE / name, PUB / "publishing_tools" / name)
    print(json.dumps({"files": total_files, "browsable_files": len(browsable), "assets": len(expected), "release_asset_bytes": sum(expected.values())}, indent=2))

if __name__ == "__main__":
    main()
