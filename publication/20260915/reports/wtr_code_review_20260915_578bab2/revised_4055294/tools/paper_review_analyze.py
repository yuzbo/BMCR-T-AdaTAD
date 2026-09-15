"""Collect full-test receipts without mixing planned, historical and public results."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from h65.paper.runtime import json_write

def normalize(source,record):
    if record.get('test_videos') not in (211,4728) or 'metrics' not in record:return None
    cfg=record.get('config',{});metric=record['metrics'].get('average_mAP')
    if metric is None:return None
    role=record.get('role','internal');seed=cfg.get('seed',3407) if role!='external_retested' else None
    ident=record.get('evaluation_id',cfg.get('id',record.get('id',record.get('variant','unknown')+'_'+record.get('backbone',''))))
    epoch=record.get('epoch',record.get('initialization',{}).get('total_epochs'))
    state=record.get('checkpoint_state','ema');parent=Path(source).parent.name
    standard=(parent==f'eval_{epoch:03}_ema' if isinstance(epoch,int) else False) or 'terminal_learned' in parent or role=='external_retested'
    return dict(run_id=ident,comparison=cfg.get('comparison',ident),dataset=cfg.get('dataset','thumos'),backbone=cfg.get('backbone',record.get('backbone','')).lower(),head=cfg.get('head','point'),
                role=role,seed=seed,epoch=epoch,state=state,standard=standard,epochs=cfg.get('epochs',80 if 'bmcr' in ident else None),
                map=100*metric,metrics=record['metrics'],gflops=record.get('gflops'),representative_gflops=record.get('representative_gflops',record.get('gflops')),
                dataset_mean_gflops=record.get('dataset_mean_gflops'),dataset_total_gflops=record.get('dataset_total_gflops'),
                compute_scope=record.get('compute_scope','historical fixed full window'),latency_ms=record.get('latency_ms'),
                source=str(source),checkpoint=record.get('checkpoint'),config=cfg,query_counts=record.get('query_counts'),
                force_plan=record.get('force_plan'),plan_distribution=record.get('plan_distribution'),test_videos=record['test_videos'])

def collect(manifest):
    rows=[];seen=set()
    for root in manifest.get('run_roots',[]):
        folder=Path(root)
        for path in folder.glob('**/completed.json'):
            record=json.loads(path.read_text(encoding='utf-8'));row=normalize(str(path),record)
            if row is not None:rows.append(row);seen.add(str(path))
        for path in folder.glob('*_bmcr_test_epoch_*/metrics.json'):
            # Completed/profile-enriched receipt and metrics describe the same test.
            if str(path.with_name('completed.json')) in seen:continue
            row=normalize(str(path),json.loads(path.read_text(encoding='utf-8')))
            if row is not None and str(path) not in seen:rows.append(row);seen.add(str(path))
    for path in manifest.get('snapshot_files',[]):
        snapshot=json.loads(Path(path).read_text(encoding='utf-8'))
        for values in snapshot.get('evidence',{}).values():
            for item in values:
                row=normalize(item['source'],item['record'])
                if row is not None and row['source'] not in seen:rows.append(row);seen.add(row['source'])
    return rows

def main(args):
    manifest=json.loads(Path(args.manifest).read_text(encoding='utf-8'));rows=collect(manifest);out=Path(args.output);json_write(out/'full_results.json',rows)
    current=[r for r in rows if r['seed']==42 or r['role']=='external_retested'];best={}
    for row in current:
        if not row['standard'] or row['state'] not in ('ema','official_ema'):continue
        key=row['run_id']
        if key not in best or row['map']>best[key]['map']:best[key]=row
    json_write(out/'best_results.json',list(best.values()))
    counts=dict(full_test_receipts=len(rows),current_full_tests=len(current),historical_full_tests=len(rows)-len(current),best_current_runs=len(best))
    report=['**完整课程与论文证据更新**','',f'当前有{len(current)}条当前协议/公开复测完整结果；历史记录{len(rows)-len(current)}条。未测配置不产生性能点。','',
            '|配置|角色|种子|最佳已测mAP %|epoch|全测试平均GFLOPs/窗|代表完整窗GFLOPs|','|---|---|---|---:|---:|---:|---:|']
    for row in sorted(best.values(),key=lambda r:(r['dataset'],r['backbone'],-r['map'])):
        report.append(f"|{row['run_id']}|{row['role']}|{row['seed']}|{row['map']:.4f}|{row['epoch']}|{row['dataset_mean_gflops']}|{row['representative_gflops']}|")
    report+=['','最佳和终点分别保留；40轮控制仅与完整模型前40轮相同候选比较。H65/BMCR/Cross是内部方法，公开reported数字不能混充本框架复测。',
             '训练账本中的forward GFLOPs是明确范围的代理，外部教师查询与完整墙钟时间另报，不将它称为完整反向训练FLOPs。',
             'Strong/Viable/Weak仅作作者内部结果归纳；所有正式课程仍按预登记轮次完成，不作为部署或早停门槛。']
    (out/'REPORT.zh.md').write_text('\n'.join(report)+'\n',encoding='utf-8');json_write(out/'summary.json',counts)
    from tools.paper_review_plot import plot_all
    plot_all(rows,out/'figures',manifest)
    print(json.dumps(counts))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);main(p.parse_args())
