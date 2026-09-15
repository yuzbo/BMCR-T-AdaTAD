"""Archive the three user reviews and adopt their bounded, current-next mini."""
import json
from pathlib import Path
import shutil
from deploy_probe import LOCAL,OUT

research=LOCAL/'research/rfv_sprint_20260915'
sources={
 'AUDIT_AND_NEXT_MINI.zh.md':Path('E:/下载/AUDIT_AND_NEXT_MINI.zh.md'),
 'USER_REVIEW_ADOPTION.zh.md':Path('C:/Users/skywalker/.codex/attachments/5fb005b4-06fb-4894-9589-e0ed92a482e2/pasted-text.txt'),
 'USER_VALUE_EVIDENCE_REVISION.zh.md':Path('C:/Users/skywalker/.codex/attachments/7e5f1d48-9f44-4401-afc7-4c51cb0a3fce/pasted-text.txt')}
for name,source in sources.items():
    for folder in (OUT/'inputs',research/'inputs'):
        folder.mkdir(exist_ok=True);destination=folder/name
        if destination.exists() and destination.read_bytes()!=source.read_bytes():raise RuntimeError('Existing input archive differs')
        shutil.copyfile(source,destination)
for filename in ('EXPERIMENT_PLAN.zh.md','EXECUTION_STATE.zh.md','EXPERIMENT_QUEUE.json'):
    current=OUT/filename;backup=OUT/(current.stem+'.before_reverse_mini'+current.suffix)
    if current.exists() and not backup.exists():shutil.copyfile(current,backup)
shutil.copyfile(research/'REVERSE_MINI_PLAN.zh.md',OUT/'EXPERIMENT_PLAN.zh.md')
shutil.copyfile(research/'REVERSE_MINI_PLAN.zh.md',research/'PROTOCOL.zh.md')
queue=json.loads((OUT/'EXPERIMENT_QUEUE.json').read_text(encoding='utf8'))
queue.update(plan='RFV-T Reverse-Consistency Mini after completed Value-R1',
    terminology='Concrete temporal frame exchanges / spatial token and depth computation choices; legacy action_* fields unchanged',
    atlas_freeze='Original S GPU measurements frozen. Separately authorized interaction_v1_s 792 windows complete; CPU AP/bootstrap finishing. B held.',
    current_next=['CPU and real GPU reverse contract','P1/P1-bi/P2 original8/8 structural mini','Saved B0 decision and margin export','Frozen D/S route and clipping diagnostics'],
    paused=['coverage common-held4 fit','partner96 representation revision','new Graph/R1 fit','all T/Graph/FVD detector courses'])
queue['scientific_sources']['value_r1']='b644d870d1845abbc1e4fd5ab7780f29ff96a53a'
queue['resources'].update(AutoDL_GPU0='R1 PID283745 completed; available subject fresh allocation check',
    AutoDL_GPU1='Atlas interaction PID284842 completed19:10:33; owner released GPU1',
    AutoDL_CPU='Atlas observer287713 finishing interaction statistics',
    A100='b644d87 deployed,14CPU tests passed; new reverse source pending exact-revision validation')
queue['selection'].update(graph_degree='v3 Static9/Dynamic5 retained from common9 candidates; no new scientific fit',
    reverse_mini='original fit8/held8; calibration descriptive; no inner10 or outer20; STOP0; three seeds2000updates')
queue['decisive_outputs']=[x for x in queue['decisive_outputs'] if x.get('file')!='VALUE_R1_GATE.json']
queue['decisive_outputs'].append(dict(file='VALUE_R1_GATE.json',status='LEARNABILITY_FAIL',
    source_revision='b644d870d1845abbc1e4fd5ab7780f29ff96a53a',within_r1_regret=.00159884234269,
    within_r0_regret=.00151919891437,within_stop_regret=.00141353726387,
    within_r1_spearman=.00539683,next='Single reverse-consistency mini; original packing-position-family OOD retained'))
(OUT/'EXPERIMENT_QUEUE.json').write_text(json.dumps(queue,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
state='''# RFV-T 当前执行状态

2026-09-15。本文件已按三份最新审阅更新。完整当前命令见 EXPERIMENT_PLAN.zh.md；这次主修改是反向一致性 mini，不是扩充 Graph/FVD 模块矩阵。

当前 scientific branch 为 codex/wtr-rfv-20260915。b644d87 的 Value-R1 已完成18个head并经独立复核，结论 LEARNABILITY_FAIL 保持。full inner10 的 regret 改善CI跨0；within R1 regret0.001598842、R0 0.001519199、STOP0.001413537，held rho0.00540。fit rho约0.944，说明它拟合了已见数据。

原 within split 的fit200全部位于tubelet局部位置5，held200为位置1/2。几何范围检查没有发现先前疑似的常量列突变；不能把失败简单归为一般同位置插值失败，也不能唯一归因于packing。三个新评论审阅的是更早6d1f924；采用建议不重写实验发生顺序。

主线：完成真实 S/S′/S 合同后，以相同407、原fit normalization、原8/8、2000updates×3seed运行P1与P2。P1-bi直接复用P1 raw权重零新增训练。shared bias在P2输出差分中相消；反向membership仍为(1,0)。反向signed distance原尺度可达−65至−75σ，将记录优化状况。覆盖held4和partner96拟合均暂缓。

并行：RISE仅导出已完成B0的共同state分数、STOP、top2 margin与选择变化，β1.1冻结，逐视频复现旧结果。D/S先定位原405的真实checkpoint并回收现有grad_norm，再做独立固定checkpoint路由/梯度诊断；不改课程、不做optimizer更新。旧四条4090课程继续，最后核实进度仍以16:40回执为准，不能把新40评测当作已完成。

AutoDL R1 PID283745已完成。Atlas owner报告interaction_v1_s全部792窗于19:10:33完成，GPU PID284842退出，两张卡均已释放，但新部署仍先读取实际占用。Atlas CPU287713统计收尾；新interaction与旧S Atlas的source分别记录。A100和4090已通过b644d87的14项CPU合同，新反向源待独立核验。

T-local-CF的cal20/46窗增量仍为+0.528413pp，视频CI[+0.082440,+0.714687]；这是有限GT辅助机会，不是廉价Value已学会。Graph旧mini不稳定，历史RISE有drift但Future未优于Post/EMA。T/Graph/FVD长训均未解锁，最终路线科学可行性尚未证明。epoch80 primary、test不选模型、outer20封存继续执行。

已发布快照保持6d1f924及旧release不变。新代码、合同、结果将以新不可变SHA逐项发布；未提交实现不能被称为已经部署或科学PASS。所有完整新实验完成后独立交叉审核。
'''
(OUT/'EXECUTION_STATE.zh.md').write_text(state,encoding='utf8')
for name in ('EXPERIMENT_QUEUE.json','EXECUTION_STATE.zh.md'):
    shutil.copyfile(OUT/name,research/name)
print('Adopted current reverse mini; archived all three user reviews; preserved R1 and old snapshots.')
