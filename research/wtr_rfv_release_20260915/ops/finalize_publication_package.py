"""Finish the evidence package and its portable public index; no experiments."""
import json
from pathlib import Path
import shutil
import zipfile
from deploy_probe import LOCAL,OUT
HUB=LOCAL/'research/wtr_rfv_release_20260915';RESEARCH=LOCAL/'research/rfv_sprint_20260915'
ASSETS=OUT/'github_release_assets';REPO='https://github.com/yuzbo/BMCR-T-AdaTAD';TAG='wtr-rfv-evidence-20260915'
manifest=json.loads((HUB/'SOURCE_MANIFEST.json').read_text())
ledger=OUT.parent/'wtr_fasttrack_20260915/FINAL_MODEL_LEDGER.csv'
for folder in (RESEARCH,OUT):
    shutil.copyfile(ledger,folder/ledger.name)
shutil.copyfile(OUT.parent/'wtr_fasttrack_20260915/ledger_preview.png',HUB/'figures/ledger_preview.png')
shutil.copyfile(OUT.parent/'wtr_fasttrack_20260915/build_ledger.mjs',HUB/'ops_build_ledger.mjs')
shutil.copyfile(Path('C:/Users/skywalker/.codex/attachments/8bef34df-73a6-448b-8540-4fed0d2d0b93/pasted-text.txt'),
    HUB/'design/USER_GLOBAL_MODEL_REQUEST.txt')
for path in OUT.glob('*.py'):
    destination=HUB/'ops'/path.name;destination.parent.mkdir(exist_ok=True);shutil.copyfile(path,destination)
raw_receipt=json.loads((OUT/'RAW_PUBLICATION_RECEIPT.json').read_text())
manifest['artifacts']=[a for a in manifest['artifacts'] if a['name'] not in (raw_receipt['name'],'rfv-evidence-and-course-logs.zip')]
manifest['artifacts'].append(raw_receipt)
target=ASSETS/'rfv-evidence-and-course-logs.zip';entries=[]
folders=[RESEARCH/'evidence',RESEARCH/'figures_reverse',HUB/'courses',HUB/'reviews',HUB/'design',HUB/'figures',HUB/'ops']
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as bundle:
    for folder in folders:
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                name=path.relative_to(LOCAL).as_posix();bundle.write(path,name)
                entries.append(dict(path=name,bytes=path.stat().st_size))
    bundle.writestr('RELEASE_CONTENTS.json',json.dumps(dict(sources=manifest['sources'],files=entries),indent=2))
manifest['artifacts'].append(dict(name=target.name,bytes=target.stat().st_size,files=len(entries),
    scope='RFV measured evidence, input identities, historical failures, code review, all currently available D/S training logs and complete milestone metadata'))
manifest['course_snapshot_as_of']=json.loads((HUB/'courses/SNAPSHOT.json').read_text())['recorded_at']
manifest['code_publication']='Main RFV Git branch plus exact-source Atlas/Raw archives; historical runtime duplicates and local resource files excluded from supplementary source archives'
(HUB/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
(OUT/'PUBLICATION_STAGING.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
lines=['# 复现、源码身份与下载','',
 '本次发布不运行新模型。主Git分支含当前RFV完整实现、配方、工具、测试、报告、图和重要数值；Atlas/Raw保持独立科学来源，以精确源码包补充。历史已发布内容继续保留。','',
 '## Release资产','',
 '|文件|大小(MiB)|内容|','|---|---:|---|']
descriptions={'atlas-s-numerical-evidence.zip':'完整S/interaction统计、AP缓存、action manifests及回执/现有日志',
 'atlas-publication-figures.zip':'已交付Atlas最终PDF/PNG/SVG、图注和精确数值表',
 'raw-existing-evidence-and-logs.zip':'只读回收的Raw已有JSON/log与protocol',
 'rfv-evidence-and-course-logs.zip':'RFV/R1/Reverse/RISE/D-S证据、审阅、绘图、所有当前可用课程日志'}
for row in manifest['artifacts']:
    desc=descriptions.get(row['name'],'对应不可变SHA的完整当前源码；私有资源/重复legacy runtime排除项列在manifest')
    lines.append(f"|[{row['name']}]({REPO}/releases/download/{TAG}/{row['name']})|{row['bytes']/2**20:.2f}|{desc}|")
lines += ['','## 不可变科学身份','','|部分|SHA|','|---|---|']
for key,value in manifest['sources'].items():lines.append(f'|{key}|{value}|')
lines += ['',
 '文档/整理提交不回填为旧运行source。源码包包含h65、tools、configs、tests、第三方源码及适用研究协议；源码包不是包含私有资源配置的完整机器镜像。主GitHub tag另外提供GitHub自动生成的Source code下载。','',
 '## 只重绘本次RFV和课程图','',
 '在所需Python环境中使用Matplotlib/NumPy，以下命令仅读已保存JSON，不加载detector、不创建作业：','',
 '~~~bash',
 'python tools/rfv_plot_publication_courses.py --snapshot research/wtr_rfv_release_20260915/courses/SNAPSHOT.json --output output/publication_courses',
 'python tools/rfv_plot_reverse.py --reverse-report research/rfv_sprint_20260915/evidence/reverse_4aa1ca2/REVERSE_MINI/REVERSE_MINI.json --rise-report research/rfv_sprint_20260915/evidence/reverse_800bcd1/RISE_CHOICES/RISE_CHOICES.json --output output/reverse',
 '~~~','',
 'Atlas跨轴数值和图可在对应Atlas源码包中配合numerical-evidence的analysis/interaction_v1_s重绘。全部已导出PDF/PNG/SVG已提供；重绘某些原Atlas视觉案例需要原始window/thumbnail记录，重新执行完整AP则需要数据、权重和对应冻结执行协议。不要用最终统计JSON冒充原始RGB/模型预测。','',
 '## 实验复现与资产','',
 'GPU重现实验应checkout其实际science SHA，使用对应数据/初始化资源和明确的bank/action manifest，而不是任意新文档HEAD。入口为tools/rfv_run.py、rfv_value_r1.py、rfv_reverse_mini.py、rfv_forecast.py及rfv_ds_diagnostic.py。保存的命令、分区、配置、normalization来源、query counts、原始失败和完成回执可直接核对。','',
 '原始视频、官方大权重、训练checkpoint和机器凭据不包含在此发布。模型训练依赖的初始化来源和实际checkpoint路径保存在运行metadata；需要自行取得相应资产。重要已有数值和日志按发布清单完整保留，未声称上传所有视频、缓存或模型权重。','',
 'ops目录是实际使用的采集/部署/归档脚本来源记录，包含原机器路径；它们不是绘图复现的自动入口。当前四条旧D/S课程仍按快照记录自身运行，RFV和Atlas自动跟进保持暂停。','',
 '## 历史记录','',
 f'[前一完整历史快照]({REPO}/releases/tag/wtr-snapshot-20260915)继续保留。history/及raw/design_history中的原文带有历史日期、旧PID和本机路径，不是当前运行命令或新模型状态。当前入口以本目录STATUS_AND_FINDINGS和METHOD_AND_IMPLEMENTATION为准。','']
(HUB/'REPRODUCE_AND_SOURCES.md').write_text('\n'.join(lines),encoding='utf8')
old_queue=json.loads((OUT/'EXPERIMENT_QUEUE.json').read_text(encoding='utf8'))
history=HUB/'history/EXPERIMENT_QUEUE.before_publication.json'
if not history.exists():history.write_text(json.dumps(old_queue,ensure_ascii=False,indent=2),encoding='utf8')
queue=dict(schema='WTR_RESEARCH_LAYERS_V2',request_scope='organize and publish; no new experimental deployment',
    research_execution='PAUSED_BY_USER',automation='PAUSED',snapshot=manifest['course_snapshot_as_of'],
    current_courses=json.loads((HUB/'COURSE_SUMMARY.json').read_text())['courses'],
    local_diagnostics=dict(R1='LEARNABILITY_FAIL',reverse='STRUCTURE_SIGNAL_FAIL',local_graph='FAIL_PRELIMINARY',
        historical_RISE='FORECAST_FAIL',ds_fixed_diagnostic='COMPLETE_REVIEW_PASS',atlas='COMPLETE',interaction='INCONCLUSIVE',raw='TECH_PASS_COVERAGE_INSUFFICIENT'),
    formal_candidates=dict(status='DESIGN_IMPLEMENTATION_PENDING',task_runs_started=False,
        sequence=['Uniform384','Global-Task','Global-Task+Value','S/D independent','Joint TSD','Optional Graph/RISE','Raw time/ROI acquisition'],
        gates='own technical and task evidence; old local-pair gates do not veto a different family'),
    sources=manifest['sources'],final_validated_model=None,publication_entry=REPO+'/releases/tag/'+TAG)
for folder in (OUT,RESEARCH):(folder/'EXPERIMENT_QUEUE.json').write_text(json.dumps(queue,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
state='# 当前执行与发布状态\n\n本次按用户要求整理并发布代码、已有结果、可视化、审阅及正式模型设计。研究自动推进仍暂停，没有启动新实验。停止交接原文保留于STOP_HANDOFF.zh.md。\n\n最新只读快照为'+manifest['course_snapshot_as_of']+'。D60与S40完整结果、R1/Reverse/RISE结论和D/S最终核验见[当前报告](../wtr_rfv_release_20260915/STATUS_AND_FINDINGS.zh.md)。正式全轴T/S/D/Raw候选与实现差距见[Method设计](../wtr_rfv_release_20260915/METHOD_AND_IMPLEMENTATION.zh.md)。\n\n旧局部gate只约束旧recipe；正式候选处于待实现状态，不能冒称已经训练。所有Final仍为no。原science身份与过去失败均保留；GitHub发布状态由Release与实际远端提交确认。\n'
(RESEARCH/'EXECUTION_STATE.zh.md').write_text(state,encoding='utf8')
scope='# 本次整理与论文模型分层\n\n研究主动推进保持暂停。本次只读回收已有进度、完成D/S结果审阅并公开归档，不启动任何正式Global课程。\n\n诊断线与正式模型线分开；旧局部pair/RISE门不约束新家族。最新方法规格、执行差距和必要自身证据见[Method与实现](../wtr_rfv_release_20260915/METHOD_AND_IMPLEMENTATION.zh.md)，真实结果见[进度](../wtr_rfv_release_20260915/STATUS_AND_FINDINGS.zh.md)。既有80轮D/S是完整组件消融，Atlas是characterization，正式全局T/S/D/Raw是待实现候选，已证明最终模型为空。\n\n原RFV注册协议仍保存在FAST_SPRINT_V3.zh.md、REVERSE_MINI_PLAN.zh.md和DS_FIXED_DIAGNOSTIC_PLAN.zh.md；它们的运行身份及历史门不重写。\n'
(RESEARCH/'PROTOCOL.zh.md').write_text(scope,encoding='utf8')
shutil.copyfile(RESEARCH/'EXECUTION_STATE.zh.md',OUT/'EXECUTION_STATE.zh.md')
shutil.copyfile(RESEARCH/'PROTOCOL.zh.md',OUT/'EXPERIMENT_PLAN.zh.md')
print(json.dumps(dict(assets=[{k:a[k] for k in ('name','bytes')} for a in manifest['artifacts']],
    total_bytes=sum(a['bytes'] for a in manifest['artifacts'])),ensure_ascii=False))
