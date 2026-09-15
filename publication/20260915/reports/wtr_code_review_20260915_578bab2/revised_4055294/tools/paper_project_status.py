"""Build the project index from captured deployment and full-test evidence (CPU only)."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def family(cfg):
    ident = cfg["id"]
    if ident.startswith("graph_"):
        return "Graph"
    if ident.startswith("native_adatad_"):
        return "Native AdaTAD"
    if cfg["dataset"] == "anet":
        return "ActivityNet"
    if cfg["backbone"] == "internvideo_mq":
        return "InternVideo1-MQ"
    if cfg["head"] == "tadtr":
        return "TadTR"
    if cfg.get("static_mode") == "pbd":
        return "PBD-style"
    if cfg.get("static_compression"):
        return "Static"
    if "full_v2" in ident:
        return "Full-V2"
    if cfg.get("family") == "state":
        return "Support / depth mechanisms"
    if cfg.get("family") == "decoder" or cfg.get("decoder") in ("mae", "tcn", "interpolate"):
        return "Decoder"
    if cfg.get("family") == "dependency":
        return "Teacher / initialization dependence"
    if cfg.get("family") == "supervision" or cfg["comparison"] in ("no_external", "no_self", "no_full_gt"):
        return "Supervision"
    if cfg["comparison"].startswith("axes_"):
        return "Independent axes"
    if cfg["comparison"] in ("full", "uniform", "dense"):
        return {"full": "Full-V1", "uniform": "Uniform Full", "dense": "Dense adaptation"}[cfg["comparison"]]
    return "Other routing / architecture controls"


def last_training_record(course):
    # The capture is a byte tail and can start/end in a partial JSONL record.
    rows = []
    for line in (course.get("train_tail") or "").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "successful_updates" in row:
            rows.append(row)
    return rows[-1] if rows else None


def main(args):
    snapshot = read(args.snapshot)
    results = read(args.results)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stages = snapshot["deployment"]["stages"]
    observed = {item["source"] for values in snapshot["evidence"].values() for item in values
                if item["record"].get("test_videos") in (211, 4728) and "metrics" in item["record"]}
    missing = observed - {row["source"] for row in results}
    if missing:
        raise RuntimeError(f"Refresh normalized full_results before publishing: {sorted(missing)}")
    lookup = {p.stem: p for folder in ("paper", "paper_review", "native_adatad", "graph")
              for p in (ROOT / "configs" / folder).glob("*.json")}
    by_config = defaultdict(list)
    for stage_id, stage in stages.items():
        if stage.get("config_id"):
            by_config[stage["config_id"]].append(dict(stage_id=stage_id, **stage))
    configurations = []
    for ident, items in sorted(by_config.items()):
        path = lookup[ident]
        cfg = read(path)
        training = next((s for s in items if s["kind"] == "train"), None)
        course = snapshot["courses"].get("train_" + ident, {})
        tests = [r for r in results if r["run_id"] in (ident, ident + "_official_direct")]
        configurations.append(dict(id=ident, family=family(cfg), config_path=path.relative_to(ROOT).as_posix(),
            config=cfg, training_status=training["status"] if training else "EVAL_ONLY",
            training_job_id=training.get("job_id") if training else None,
            last_training_record=last_training_record(course), full_test_count=len(tests),
            stages=items, results=tests))
    registered = snapshot["graph_registration"]
    trains = [s for s in stages.values() if s["kind"] == "train"]
    actual = (len(configurations), len(trains), len(stages))
    expected = tuple(registered[key] for key in ("total_configurations", "total_training_courses", "total_stages"))
    if actual != expected:
        raise RuntimeError(f"Registration differs from captured stages: {actual} versus {expected}")
    current = [r for r in results if r["seed"] == 42 or r["role"] == "external_retested"]
    summary = dict(time=snapshot["time"], configurations=len(configurations), training_courses=len(trains),
        stages=len(stages), stage_status=dict(Counter(s["status"] for s in stages.values())),
        stage_kind=dict(Counter(s["kind"] for s in stages.values())),
        training_status=dict(Counter(s["status"] for s in trains)),
        full_test_receipts=len(results), current_full_tests=len(current), historical_full_tests=len(results)-len(current),
        controller_pid=snapshot["deployment"]["controller_pid"], controller_alive=snapshot["controller_alive"],
        sources=dict(snapshot=Path(args.snapshot).resolve().relative_to(ROOT).as_posix(), normalized_results=Path(args.results).resolve().relative_to(ROOT).as_posix()))
    write(out / "summary.json", summary)
    write(out / "experiment_index.json", configurations)
    write(out / "current_results.json", current)
    write(out / "stage_index.json", stages)
    lines = ["# 全部已注册配置", "", f"服务器快照：{snapshot['time']}。配置数不是已完成实验数。", "",
             "WAITING=持久调度队列；PENDING=已提交Slurm等待资源；RUNNING=当前执行；EVAL_ONLY=仅评测配置。",
             "当前训练状态仅表示该课程的终点状态。Full-V1已完成部分训练后保存断点等待续跑，不是零进度。", "",
             "|组|配置|轮次|训练状态|训练作业|当前完整测试数|", "|---|---|---:|---|---|---:|"]
    for row in sorted(configurations, key=lambda c: (c["family"], c["id"])):
        epoch = row["config"].get("epochs", "—") if row["training_status"] != "EVAL_ONLY" else "仅评测"
        lines.append(f"|{row['family']}|[{row['id']}](../../{row['config_path']})|{epoch}|{row['training_status']}|{row['training_job_id'] or '—'}|{row['full_test_count']}|")
    lines += ["", "生成器以主Full-V1的同epoch40结果代表T1D1S1，另登记七个axes课程。主模型与axes控制的动态预算设置不同，不能据此宣称已具备严格同策略的八格因子矩阵；论文仍需核对最终候选与固定策略参照的匹配。当前也尚无七格的独立完整结果。",
              "P00为Static，P01为PBD-style，不能按配置名是否包含pbd/static判断是否已注册。原版AdaTAD的K768两条只评测；K384两条另有80轮训练。",
              "", "## 按组统计", "", "|组|配置数|训练课程|训练状态|", "|---|---:|---:|---|"]
    groups = defaultdict(list)
    for row in configurations:
        groups[row["family"]].append(row)
    for label, rows in sorted(groups.items()):
        counts = Counter(r["training_status"] for r in rows if r["training_status"] != "EVAL_ONLY")
        lines.append(f"|{label}|{len(rows)}|{sum(counts.values())}|{dict(counts)}|")
    (out / "EXPERIMENTS.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lines = ["# 全部部署阶段", "", f"服务器快照：{snapshot['time']}。完整参数、依赖、输出路径与重试记录见[stage_index.json](stage_index.json)。", "",
             "训练内的10/20/40/60/80 EMA评测不一定各占一个调度stage；完整测试次数应查结果回执。", ""]
    for kind in sorted({s["kind"] for s in stages.values()}):
        lines += [f"## {kind}", "", "|stage|状态|当前作业|优先级|", "|---|---|---|---:|"]
        for stage_id, s in stages.items():
            if s["kind"] == kind:
                lines.append(f"|{stage_id}|{s['status']}|{s.get('job_id') or '—'}|{s.get('priority', '—')}|")
        lines.append("")
    (out / "STAGES.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
