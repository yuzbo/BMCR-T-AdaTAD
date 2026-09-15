"""Static inventory of saved command files; never imports or executes a launcher."""
import ast
import collections
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent
OUT=ROOT/'ds3_20260912/command_audit_20260912'
SCOPES={
 'legacy_root':PROJECT,
 'publication_legacy':PROJECT/'publication/BMCR-T-AdaTAD',
 'fidelity_completed':PROJECT/'fidelity_20260911',
 'mainline_draft':PROJECT/'mainline_20260911',
 'active_ds3':ROOT,
}
PACK1=PROJECT/'phase2_20260910/inputs/reference_pack/H65_agents_pack'
PACK2=PROJECT/'reviews/20260912_ds3/bundle/H65_DS3_research_20260911'
records={}
ROLES={
 'full_train.py':'H65/BMCR训练入口；配方与CLI细节依源码版本区分',
 'full_audit.py':'训练来源128窗口反事实/贡献效用审计与尺度校准',
 'full_eval.py':'完整测试与profile；早期版本只终点，当前版本支持milestone',
 'full_dispatch.py':'旧Phase2的16阶段调度；已完成的历史控制器',
 'full_compare.py':'旧Phase2结果比较，不能直接代替修正BMCR的选模/汇总',
 'preflight_full.py':'旧完整实验的S/B真实GPU预检',
 'check_full_cpu.py':'旧正式路径CPU机制检查',
 'gpu_probe.py':'初期单视频/两窗口GPU诊断',
 'one_swap_audit.py':'初期固定K单帧交换诊断',
 'check_tia_connectivity.py':'非零Adapter跨clip连接与跨视频隔离检查',
 'inspect_runtime.py':'资源/配置/模型运行环境检查',
 'verify_saved_probe.py':'初期已保存权重严格重载核验',
 'final_report.py':'初期诊断报告生成',
 'prepare_publication.py':'旧实验公开仓库整理工具',
 'run_gpu_probe.sh':'初期4090探针的站点sbatch包装器，历史已完成',
 'run_full.sh':'训练脚本包装器；根目录为旧站点版，工作树内为通用模板',
 'summarize_audit.py':'效用审计结果整理',
 'verify_results.py':'已生成实验记录复核',
 'fidelity_dispatch.py':'修正H65专用24阶段调度；没有修正BMCR注册',
 'fidelity_select.py':'修正H65总25..60测试峰值选择',
 'fidelity_progress.py':'修正H65中间结果汇总',
 'fidelity_finalize.py':'修正H65最终证据整理',
 'fidelity_compare.py':'修正H65峰值/终点与旧/官方结果比较',
 'ds3_site.py':'建立本项目数据、官方TAD、MobileNet资源链接',
 'ds3_preflight.py':'S/B dense/compact/稀疏MLP/冻结/真实更新/EMA GPU预检',
 'ds3_train.py':'80轮D1辅助训练，pilot100步属于同一轨迹',
 'ds3_eval.py':'单policy全211视频评测与三个profile场景',
 'ds3_select.py':'T24A总5..80每5轮的峰值选模',
 'ds3_dispatch.py':'当前73阶段调度与资源上限管理',
 'retrospective_20260912.py':'历史完整结果回顾与图表生成',
 'build_research_context_20260912.py':'固定提交外部研究上下文生成',
 'build_command_catalog_20260912.py':'本次静态命令索引生成，不运行实验',
 'run_job.sh':'站点实际执行包装器；须按所在实验目录区分',
 'setup_resources.py':'历史修正H65资源链接准备',
 'resources.example.sh':'资源环境变量示例，需实际站点配置',
 'independent_checks.py':'外部报告纯算术检查，不是模型训练/独立mAP复测',
 'checks.py':'第二份外部报告纯算术检查，不是模型训练/独立mAP复测',
 'source_audit.py':'原DS3包的旧锚点/源码来源审计',
 'run_experiment.py':'原DS3包的manifest启动框架；不自带模型实现',
 'verify_bundle.py':'原DS3包文件/语法/数学参考检查',
 'run_agent.sh':'原DS3包Codex任务书启动器',
 'bootstrap.sh':'原DS3包旧锚点worktree准备器',
 'run_ds3.sbatch':'原DS3包站点模板，不是当前已部署run_job.sh',
 'bootstrap_sources.sh':'原H65包获取固定旧来源的脚本',
 'check_sources.py':'原H65包来源检查',
 'run_research_agents.sh':'原H65包六个研究任务启动器',
 'run_approved_implementation.sh':'原H65包未来M0–M2实现模板入口',
 'reference_ops.py':'原DS3包数学参考实现，不是当前模型执行器',
 'test_reference_ops.py':'原DS3包数学参考测试，不是实际4090测试',
 'score42_recovery.sh':'历史65+恢复训练脚本快照，不属于当前新训练入口',
}

def add(path,scope,base,kind=None):
 if not path.is_file() or '__pycache__' in path.parts:return
 path=path.resolve();key=str(path)
 if key in records:return
 suffix=path.suffix.lower()
 kind=kind or ('python_script' if suffix=='.py' else 'shell_launcher' if suffix in ('.sh','.sbatch','.ps1','.bat','.cmd') else 'command_document')
 r=dict(scope=scope,path=path.as_posix(),relative_path=path.relative_to(base.resolve()).as_posix(),kind=kind,
        role=ROLES.get(path.name,''),cli_switches=[],has_main_guard=False)
 if suffix=='.py':
  tree=ast.parse(path.read_text(encoding='utf-8-sig'))
  r['summary']=(ast.get_docstring(tree) or '').split('\n')[0]
  for node in ast.walk(tree):
   if isinstance(node,ast.Constant) and node.value=='__main__':r['has_main_guard']=True
   if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument':
    options=[a.value for a in node.args if isinstance(a,ast.Constant) and isinstance(a.value,str)]
    r['cli_switches'].extend(options)
  r['cli_switches']=list(dict.fromkeys(r['cli_switches']))
 records[key]=r

for scope,base in SCOPES.items():
 for folder in (base/'tools',base/'scripts',base/'site',base/'fidelity_20260911/site',base/'ds3_20260912/site',base/'phase3_20260911/site'):
  if folder.is_dir():
   for p in folder.rglob('*'):
    if p.suffix.lower() in ('.py','.sh','.sbatch','.ps1','.bat','.cmd'):add(p,scope,base)
 add(base/'resources.example.sh',scope,base,'resource_template')
 for relative in ('EXPERIMENT_PLAN.md','STATE.md','phase2_20260910/PLAN.md','phase2_20260910/deployment.json',
                  'fidelity_20260911/PLAN.md','fidelity_20260911/DEPLOYMENT_20260911.md','fidelity_20260911/deployment.json',
                  'ds3_20260912/PLAN.md','ds3_20260912/DEPLOYMENT.md','phase3_20260911/PLAN.md','bmcr_fidelity_20260912/README.md'):
  add(base/relative,scope,base,'experiment_plan_or_receipt')
 if (base/'docs').exists():
  for p in (base/'docs').rglob('*'):
   if p.is_file() and 'prompt' in p.name.lower():add(p,scope,base,'research_prompt')

for scope,base in (('original_H65_pack',PACK1),('original_DS3_pack',PACK2)):
 for p in base.rglob('*'):
  if not p.is_file():continue
  if p.suffix in ('.py','.sh','.sbatch'):add(p,scope,base)
  elif p.parent.name in ('prompts','agents'):add(p,scope,base,'agent_task_book')
  elif p.name in ('README.md','README_zh.md','COMMANDS_zh.md','TRAINING_HANDOFF.md','AGENTS.md','cli_v1.md'):
   add(p,scope,base,'command_document')
  elif p.name in ('experiments.json','SOURCE_LOCK.json','experiment_manifest.example.json','implementation.example.json','launch.template.json','gate.template.json','route_v1.json'):
   add(p,scope,base,'plan_or_manifest_template')

for label,name in (('BMCR_review1','20260911_external_bmcr_review'),('BMCR_review2','20260911_external_bmcrt_review_v2')):
 base=PROJECT/'reviews'/name
 for p in base.rglob('*.py'):add(p,label,base,'review_check_script')
base=PROJECT/'diagnostics/h65_65_gap_20260911'
for p in base.rglob('*'):
 if p.suffix in ('.py','.sh'):add(p,'historical_diagnosis',base,'diagnosis_or_source_snapshot')
for sub,label in (('upstream/tools','upstream_tools'),('references/ASFormer','ASFormer_reference')):
 base=PROJECT/sub
 for p in base.rglob('*'):
  if p.suffix in ('.py','.sh','.sbatch') and p.name!='__init__.py':add(p,label,base,'dependency_tool_or_module')
for base,label in ((PROJECT,'legacy_tests'),(ROOT,'active_tests'),(SCOPES['mainline_draft'],'draft_tests')):
 for p in (base/'tests').glob('test_*.py'):add(p,label,base,'test_suite_source')
add(PROJECT.parent/'handoff_4090.md','resource_handoff',PROJECT.parent,'resource_command_document')
automation=Path('C:/Users/skywalker/.codex/automations/h65-ds3-80/automation.toml')
add(automation,'current_heartbeat',automation.parent,'automation_configuration')

records=sorted(records.values(),key=lambda r:(r['scope'],r['relative_path']))
live=json.loads((OUT/'live_deployment.json').read_text())
stages=live['deployment']['stages'];counts=dict(collections.Counter(s.get('status','UNKNOWN') for s in stages.values()))
groups=collections.defaultdict(list)
for r in records:groups[r['scope']].append(r)
payload=dict(scope='H65 project saved command/script/task/template files and named current heartbeat; model libraries only in separately marked dependency/source categories. Inline shell history is not a command file.',
             snapshot_time=live['timestamp'],files=records,physical_file_instances=len(records),counts_by_scope={k:len(v) for k,v in groups.items()},
             duplicate_note='Worktree copies are separate file instances, not extra experiments; same basename may have different CLI and recipe.',
             ds3_registered_stage_count=len(stages),ds3_status_counts=counts,ds3_stages=stages,
             current_heartbeat=dict(id='h65-ds3-80',status='ACTIVE',interval_minutes=30))
(OUT/'command_inventory.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def link(path,label):
 p=Path(path)
 target=os.path.relpath(p,OUT).replace('\\','/') if p.is_relative_to(ROOT) else p.as_posix()
 return f'[{label}](<{target}>)'

lines=['**保存的命令文件、任务书与模板逐文件索引**','',
       f"本次登记{len(records)}份物理文件实例，按来源区分。包含脚本、任务书、模板、测试及单列的依赖工具；不等于{len(records)}项实验。",'',
       '当前活动根目录为`h65_clean_adatad/ds3_20260912`。外部原件与暂停草稿的绝对链接只在本地有效，原件内容未因此公开上传。',
       '参数为静态提取，不导入或执行脚本；无参数列表不代表已验证可直接运行。同名脚本在不同工作树中可能采用不同配方。','']
for scope,items in groups.items():
 lines.extend([f'**{scope}：{len(items)}份。**','', '|文件|类别/作用|源码声明的参数|','|---|---|---|'])
 for r in items:
  flags=' '.join(r['cli_switches']) or '无argparse参数/见源码'
  role=r['role'] or r.get('summary') or r['kind']
  lines.append('|'+link(r['path'],r['relative_path'])+'|'+role.replace('|','/')+'|`'+flags.replace('|','/')+'`|')
 lines.append('')
(OUT/'FILES.zh.md').write_text('\n'.join(lines).rstrip()+'\n',encoding='utf-8')

lines=['**当前DS3调度器登记的全部阶段**','',f"快照：{live['timestamp']}。共{len(stages)}阶段；状态{counts}。WAITING代表依赖/资源尚未满足，不等于模型没实现。",'',
       '以下只是从收据提取的入口，不是本次重跑命令。GPU入口由本项目站点包装器与现有控制器安排，不能在登录节点直接跑，也不应启动第二个控制器。','',
       '|阶段|种类|状态/作业|入口或待解析模板|依赖|','|---|---|---|---|---|']
for name,s in stages.items():
 args=s.get('args')
 if args:command=' '.join(args)
 elif s.get('kind') in ('selected_test','selected_profile'):
  command=f"tools/ds3_eval.py --backbone {s['backbone']} --policy {s.get('policy','T24A')} --epoch <T24A选中epoch>"+(' --profile-only' if s['kind']=='selected_profile' else '')
 else:command='尚未解析'
 lines.append('|'+name+'|'+s['kind']+'|'+s.get('status','UNKNOWN')+'/'+str(s.get('job_id','—'))+'|`'+command+'`|'+', '.join(s['dependencies'])+'|')
(OUT/'REGISTERED_STAGES.zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(files=len(records),groups={k:len(v) for k,v in groups.items()},ds3_stages=len(stages),states=counts),ensure_ascii=False))
