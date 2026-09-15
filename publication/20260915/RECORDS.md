# 实验记录与完整源码下载

本次快照包含 **7,767 个原始实验文件**，解压后约 **28.04 GB**；日志与测量记录压缩后约 **2.34 GB**，共 70 个独立 tar.gz 分卷。另附三条源码分支及完整提交历史的 Git bundle（115.6 MB）。大小使用十进制单位。

[Release 全部附件](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/tag/wtr-snapshot-20260915) · [可浏览的小型日志、配置和回执](records) · [逐文件清单](manifests)

## 采集范围与时间

下表时间为北京时间（UTC+8）。开始时枚举快照文件范围，随后逐文件复制；实验始终继续运行。每个 manifest 的 `captured_utc` 是逐文件复制时间，`mtime_ns` 是初始枚举元数据，`bytes` 是实际归档大小。采集开始后新增的文件不属于这次快照；较晚复制的现有日志可能包含后续追加内容。

| 服务器 | 文件数 | 原始 GB | 压缩 MB | 分卷数 | 采集开始 | 采集结束 | 清单 |
|---|---:|---:|---:|---:|---|---|---|
| 4090 | 1,086 | 2.285 | 216.05 | 6 | 2026-09-15 12:14:58 | 2026-09-15 12:16:42 | [4090 manifest](manifests/4090-manifest.json) |
| a100 | 23 | 0.000 | 0.01 | 1 | 2026-09-15 12:15:06 | 2026-09-15 12:15:06 | [a100 manifest](manifests/a100-manifest.json) |
| atlas | 6,658 | 25.756 | 2123.25 | 63 | 2026-09-15 12:14:57 | 2026-09-15 12:26:37 | [atlas manifest](manifests/atlas-manifest.json) |

4090 范围包含新 Core、Raw 初版与 27d557e 修正版，以及 paper、support、Graph 旧课程的训练、评测、失败、取消和部署记录。A100 保留课程排队、配置、部署与迁移回执。Atlas 包含 `results/` 逐窗口记录、`analysis/`、`logs/`、`queue/` 和 `output/`。完整路径和排除项见 manifest。

模型权重、checkpoint、视频与帧数据、依赖环境、密钥、缓存字节码、已有重复打包文件未上传。源码单独通过 Git 分支和 bundle 发布。所有已选记录保持原始内容；独立 ZIP 设计原件也保存在报告目录。

## 常用浏览入口

- [Core D-U 完整训练日志](records/4090/core/research/paper/runs/wtr_d_u_s42/train.jsonl)
- [Core D-V 完整训练日志](records/4090/core/research/paper/runs/wtr_d_v_s42/train.jsonl)
- [Core 日志、评测和预检回执](records/4090/core)
- [Raw-v1 验证、mini-bank 和诊断](records/4090/raw)
- [A100 部署与迁移](records/a100)
- [Atlas 可浏览记录](records/atlas)

Git 中额外提供 1,043 份可浏览记录：通常是小于 750 KB 的日志、JSON、脚本与文本，并包含两条当前 Core 完整训练日志。大型逐窗口结果和预测输出保存在 Release 分卷中。

## 一次下载完整源码

[wtr-core-raw-atlas-source.bundle](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/wtr-core-raw-atlas-source.bundle) 保留 Core、Raw、Atlas 三条独立分支及其 Git 历史。解包不需要连接 GitHub：

```bash
git clone --branch codex/wtr-fasttrack wtr-core-raw-atlas-source.bundle wtr-source
cd wtr-source
git switch codex/wtr-raw-v1
# git switch codex/wtr-characterization-20260915
```

该 bundle 对应 Core `6e2fc7f`、Raw `27d557e`、Atlas `df2b168`。发布报告和图也可从本 Release 自动提供的 Source code ZIP 下载。

## 下载与解压原始记录

各分卷都是独立的 tar.gz 文件，不能拼接。可只下载所需服务器；同一服务器的分卷解压到同一目录。以下示例恢复 4090 记录：

```bash
gh release download wtr-snapshot-20260915 --repo yuzbo/BMCR-T-AdaTAD --pattern '4090-*' --dir records-4090
mkdir -p snapshot-4090
for part in records-4090/*.tar.gz; do
  tar -xzf "$part" -C snapshot-4090
done
```

Atlas 用 `--pattern 'atlas-*'` 并解压到独立目录；A100 同理。Atlas 测量文件恢复后可用首页的命令重建图。分卷大小按 [GitHub Release 文件规则](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)控制。

| 文件 | 压缩 MB | 文件数 |
|---|---:|---:|
| [4090-experiment-records-part01.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/4090-experiment-records-part01.tar.gz) | 37.79 | 408 |
| [4090-experiment-records-part02.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/4090-experiment-records-part02.tar.gz) | 35.54 | 53 |
| [4090-experiment-records-part03.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/4090-experiment-records-part03.tar.gz) | 39.33 | 354 |
| [4090-experiment-records-part04.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/4090-experiment-records-part04.tar.gz) | 34.52 | 48 |
| [4090-experiment-records-part05.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/4090-experiment-records-part05.tar.gz) | 37.99 | 69 |
| [4090-experiment-records-part06.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/4090-experiment-records-part06.tar.gz) | 30.87 | 154 |
| [a100-experiment-records-part01.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/a100-experiment-records-part01.tar.gz) | 0.01 | 23 |
| [atlas-experiment-records-part01.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part01.tar.gz) | 29.57 | 44 |
| [atlas-experiment-records-part02.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part02.tar.gz) | 29.94 | 40 |
| [atlas-experiment-records-part03.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part03.tar.gz) | 29.51 | 40 |
| [atlas-experiment-records-part04.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part04.tar.gz) | 30.08 | 40 |
| [atlas-experiment-records-part05.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part05.tar.gz) | 29.70 | 41 |
| [atlas-experiment-records-part06.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part06.tar.gz) | 30.27 | 40 |
| [atlas-experiment-records-part07.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part07.tar.gz) | 29.85 | 40 |
| [atlas-experiment-records-part08.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part08.tar.gz) | 29.58 | 40 |
| [atlas-experiment-records-part09.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part09.tar.gz) | 29.93 | 40 |
| [atlas-experiment-records-part10.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part10.tar.gz) | 29.76 | 40 |
| [atlas-experiment-records-part11.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part11.tar.gz) | 29.71 | 40 |
| [atlas-experiment-records-part12.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part12.tar.gz) | 30.32 | 40 |
| [atlas-experiment-records-part13.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part13.tar.gz) | 29.48 | 40 |
| [atlas-experiment-records-part14.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part14.tar.gz) | 29.77 | 40 |
| [atlas-experiment-records-part15.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part15.tar.gz) | 29.65 | 40 |
| [atlas-experiment-records-part16.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part16.tar.gz) | 29.62 | 40 |
| [atlas-experiment-records-part17.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part17.tar.gz) | 29.73 | 40 |
| [atlas-experiment-records-part18.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part18.tar.gz) | 29.93 | 40 |
| [atlas-experiment-records-part19.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part19.tar.gz) | 29.63 | 40 |
| [atlas-experiment-records-part20.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part20.tar.gz) | 29.81 | 42 |
| [atlas-experiment-records-part21.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part21.tar.gz) | 30.00 | 41 |
| [atlas-experiment-records-part22.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part22.tar.gz) | 29.67 | 40 |
| [atlas-experiment-records-part23.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part23.tar.gz) | 29.24 | 40 |
| [atlas-experiment-records-part24.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part24.tar.gz) | 29.52 | 40 |
| [atlas-experiment-records-part25.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part25.tar.gz) | 28.74 | 40 |
| [atlas-experiment-records-part26.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part26.tar.gz) | 29.59 | 40 |
| [atlas-experiment-records-part27.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part27.tar.gz) | 30.11 | 40 |
| [atlas-experiment-records-part28.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part28.tar.gz) | 28.68 | 40 |
| [atlas-experiment-records-part29.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part29.tar.gz) | 29.30 | 40 |
| [atlas-experiment-records-part30.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part30.tar.gz) | 29.35 | 40 |
| [atlas-experiment-records-part31.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part31.tar.gz) | 29.04 | 40 |
| [atlas-experiment-records-part32.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part32.tar.gz) | 29.66 | 40 |
| [atlas-experiment-records-part33.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part33.tar.gz) | 28.94 | 40 |
| [atlas-experiment-records-part34.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part34.tar.gz) | 29.56 | 40 |
| [atlas-experiment-records-part35.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part35.tar.gz) | 29.11 | 40 |
| [atlas-experiment-records-part36.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part36.tar.gz) | 29.84 | 40 |
| [atlas-experiment-records-part37.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part37.tar.gz) | 29.42 | 40 |
| [atlas-experiment-records-part38.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part38.tar.gz) | 29.29 | 40 |
| [atlas-experiment-records-part39.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part39.tar.gz) | 29.50 | 40 |
| [atlas-experiment-records-part40.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part40.tar.gz) | 29.68 | 44 |
| [atlas-experiment-records-part41.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part41.tar.gz) | 29.78 | 40 |
| [atlas-experiment-records-part42.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part42.tar.gz) | 29.79 | 40 |
| [atlas-experiment-records-part43.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part43.tar.gz) | 29.96 | 40 |
| [atlas-experiment-records-part44.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part44.tar.gz) | 29.27 | 40 |
| [atlas-experiment-records-part45.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part45.tar.gz) | 30.02 | 40 |
| [atlas-experiment-records-part46.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part46.tar.gz) | 29.87 | 40 |
| [atlas-experiment-records-part47.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part47.tar.gz) | 29.88 | 40 |
| [atlas-experiment-records-part48.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part48.tar.gz) | 29.33 | 40 |
| [atlas-experiment-records-part49.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part49.tar.gz) | 29.67 | 40 |
| [atlas-experiment-records-part50.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part50.tar.gz) | 29.76 | 40 |
| [atlas-experiment-records-part51.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part51.tar.gz) | 30.37 | 40 |
| [atlas-experiment-records-part52.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part52.tar.gz) | 29.44 | 40 |
| [atlas-experiment-records-part53.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part53.tar.gz) | 30.09 | 40 |
| [atlas-experiment-records-part54.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part54.tar.gz) | 30.07 | 40 |
| [atlas-experiment-records-part55.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part55.tar.gz) | 30.11 | 40 |
| [atlas-experiment-records-part56.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part56.tar.gz) | 30.09 | 40 |
| [atlas-experiment-records-part57.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part57.tar.gz) | 30.09 | 40 |
| [atlas-experiment-records-part58.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part58.tar.gz) | 30.10 | 40 |
| [atlas-experiment-records-part59.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part59.tar.gz) | 27.04 | 41 |
| [atlas-experiment-records-part60.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part60.tar.gz) | 37.19 | 1057 |
| [atlas-experiment-records-part61.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part61.tar.gz) | 126.10 | 1612 |
| [atlas-experiment-records-part62.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part62.tar.gz) | 162.48 | 1295 |
| [atlas-experiment-records-part63.tar.gz](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-snapshot-20260915/atlas-experiment-records-part63.tar.gz) | 48.67 | 321 |

每卷已核对 gzip 完整性、逐文件路径与大小和 manifest 一致性；GitHub 返回的附件名称、大小和 uploaded 状态也已核对。科学结论沿用相应实验报告的边界，发布核验不替代科学验证。
