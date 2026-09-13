# ActivityNet val分片来源与既有完整性证据核查

2026-09-14，只读核查。未下载任何视频/归档内容，未计算新的本地摘要，未提交、重启或取消作业。目录元数据核验时间为02:52:34 UTC+8。

**独立核查结论：14个val分片的文件名、顺序和大小均正确，但原下载闭环没有内容一致性验证，不能凭大小判定压缩流完好。** 随后协调方完成发布者LFS校验，报告仅`.09`不匹配、其余13片匹配，并已提交1288754修复；该内容定位来自协调方的1288738诊断，本任务没有重复运行它。

## 1. 精确文件与顺序

目录：`/data/run01/sczc063/yuzibo/bcr_tad_v3_implementation/activitynet/downloads`。

- 依次为`v1-2_val.tar.gz.00`至`.13`，目录元数据快照中没有额外匹配项或part。
- `.00`至`.12`每个4,194,304,000 bytes；`.13`为2,704,037,512 bytes；共57,229,989,512 bytes。14项实际大小均与公开清单匹配。
- 原始来源为[YimuWang/ActivityNet](https://huggingface.co/datasets/YimuWang/ActivityNet/tree/main)。原下载器使用该仓库的`resolve/main/<文件名>`接口。
- 对应`.09`的原始下载地址为`https://huggingface.co/datasets/YimuWang/ActivityNet/resolve/main/v1-2_val.tar.gz.09`。本次没有请求这个归档的数据体。
- 完整清单是`yimuwang_tree_20260912.json`中的43项；`remaining_archives_20260913.json`只有30项，val仅`.04`至`.13`，不适合作为完整拼接清单。

本次逐文件元数据：[anet_val_metadata_audit_20260914.json](independent_sources/anet_val_metadata_audit_20260914.json)。规范逐项清单：[anet_archive_manifest_20260913.json](independent_sources/anet_archive_manifest_20260913.json#L177)。这是修复启动前的快照，后续保留损坏备份可能增加文件，不能用本快照否认新的备份存在。

## 2. 原来实际有什么验证

| 已有证据 | 能说明什么 | 不能说明什么 |
|---|---|---|
| 文件大小、PRESENT/DOWNLOADED回执 | 达到公开字节数、断点传输已结束 | 内容与源一致、gzip完整 |
| train.00的1MiB Range/gzip头探针、train.09的4096字节续传探针 | 部分HTTP 206/Content-Range和续传请求行为正确 | val14片完整内容通过验证 |
| 11个人工小分片、2个成员的读取器PASS回执 | 标准gzip/tar能跨这些人工边界读取 | 实际57.23GB val流、视频解码或实际大文件CRC通过 |
| HF清单中的lfs.oid/xetHash | 发布者提供内容标识元数据 | 本地已经和它们比对 |

人工读取器测试确有记录：[anet_archive_reader_check.json](independent_sources/anet_archive_reader_check.json#L1)，明确`is_video_decode_check=false`。不能写成“没有任何读取器测试”，也不能扩大为真实val归档已经验过。

原下载器的检查以大小和Range为主，没有下载完成后的逐片源内容比对或完整gzip校验步骤。[下载器](independent_sources/download_anet_archives.py#L56)、[原HF元数据](independent_sources/yimuwang_tree_20260912.json#L681)

发布者README还在“H5 sum”节列出合并归档的校验值；它是公开参考信息，不是本地校验PASS回执，也不能直接定位单个坏片。[发布者README](https://huggingface.co/datasets/YimuWang/ActivityNet/blob/main/README.md)

因此，旧记录无法单独指定`.09`，也不能用某片的broken-pipe次数猜测损坏位置。最新应以协调方的14片LFS比对为准：仅`.09`内容不匹配。它定位了待修复文件，但没有单独证明字节差异具体发生在哪一次传输或存储环节。

## 3. 具体替代入口

| 来源 | 具体链接/文件 | 本次核实与限制 |
|---|---|---|
| 同一发布者YimuWang的外链 | [百度网盘](https://pan.baidu.com/s/1Ic6wrvOUJARgnqRSk-BzjA)，公开提取码`f766` | README明确给出；本次网页工具无法打开网盘页，未核实有效性、内部文件或内容独立性，不能称完好的独立副本。[来源](https://huggingface.co/datasets/YimuWang/ActivityNet/blob/main/README.md) |
| OpenTAD/ActivityNet作者授权渠道 | [ActivityNet视频申请表](https://docs.google.com/forms/d/e/1FAIpQLSdxhNVeeSCwB2USAfeNWCaI9saVT6i2hpiiizVYfa3MsTyamg/viewform) | 可读到申请页；说明处理后提供Google Drive访问、7天过期。本次未提交、未获取权限。[表单](https://docs.google.com/forms/d/e/1FAIpQLSdxhNVeeSCwB2USAfeNWCaI9saVT6i2hpiiizVYfa3MsTyamg/viewform) |
| 官方端到端实验用处理包 | `Anet_videos_15fps_short256.zip`，见[OpenTAD固定版本说明](https://github.com/sming256/OpenTAD/blob/346d09d19e2091372cec48172dbe40f7b28bdee6/tools/prepare_data/activitynet/README.md) | 作者明确说明该包已加到上述授权文件夹，15fps/短边256，所有端到端ANet实验使用它；未找到无需授权的独立公开直链，模型权重/预提取特征链接不能代替它 |
| ETAD原作者提供的同规格视频渠道 | [ETAD README](https://github.com/sming256/ETAD#to-reproduce-our-results-on-activitynet-13)，公开联系`shuming.liu@kaust.edu.sa`；另有[其链接的申请表](https://docs.google.com/forms/d/e/1FAIpQLSeKaFq9ZfcmZ7W0B0PbEhfbTHY41GeEgwsa7WobJgGUhn4DTQ/viewform) | README明确可按许可取得15fps/短边256视频；本次仅查看说明/表单，没有发送邮件或申请 |
| 第三方HF候选，非官方/非YimuWang原发布者 | [shajiayu1/Activitynet文件树](https://huggingface.co/datasets/shajiayu1/Activitynet/tree/main)，val为`v1-2_val.tar.gz_aa`至`_an` | [README.txt](https://huggingface.co/datasets/shajiayu1/Activitynet/blob/main/README.txt)说明4GB分片及拼接；页面有14片，未验证内容是否与YimuWang独立或是否完好，不能仅凭相同大小跨来源混合分片 |
| 另一个研究发布者的原视频候选 | [VideoMind ActivityNet目录](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/activitynet)，`videos.tar.gz.00`至`.11` | 目录及durations.json可见；未下载/解压验证，不称ActivityNet官方副本。目录中的3fps包不能代替当前15fps输入协议 |

没有找到“无需申请、已核实可下载且已验证完好的官方独立val副本”。**当前已经定位单个`.09`的本地内容不匹配，协调方按原发布者文件修复最直接；上述入口作为来源备选，不构成本任务另起下载的指令。**

## 4. 当前故障与修复归属

1288466在val解压时报`zlib invalid distance code` / `tarfile.ReadError: invalid compressed data`，退出1:0。02:42本任务只读统计成功journal为训练9,062、验证3,639，共12,701；逐视频FAILED行0，不能由此将整个作业判成功。

协调方随后报告：1288738已完成14片LFS校验；新作业1288754在同一allocation中修复`.09`、保留损坏原片并从journal恢复准备。新的权威回执为`/data/run01/sczc063/yuzibo/h65_clean_adatad_20260910/paper_20260913/research/paper/data_preparation_job.json`；详细恢复证据由协调方维护在其`recovery_20260914_0240`目录。本任务未重复校验内容、下载或操作作业，DST继续HOLD。
