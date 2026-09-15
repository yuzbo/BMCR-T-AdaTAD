"""Training-only actual RGB swaps, feature repair proxy, and learned S0→S1 audit."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]
from h65.frame.runtime import EXP,json_write,read_config

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--resources',default=str(EXP/'resources.local.json'));p.add_argument('--windows',type=int,default=32);p.add_argument('--pairs',type=int,default=2)
    p.add_argument('--output',required=True);p.add_argument('--dry-run',action='store_true');a=p.parse_args();cfg=read_config(a.config)
    if a.dry_run:print(json.dumps(dict(config=cfg['id'],windows=a.windows,pairs=a.pairs,actual_rgb_reencode=True,gpu_execution=False)));return
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from scipy.stats import spearmanr
    from opentad.datasets.builder import build_dataset,collate
    from h65.full.runtime import initialize_gpu,to_gpu
    from h65.frame.runtime import read_resources,data_config
    from h65.frame.model import FrameModel
    from h65.frame.utility import paired_interventions,components
    from h65.frame.contracts import EnginePolicy
    hardware=initialize_gpu();mc=data_config(cfg['backbone']);model=FrameModel(mc.model,cfg,read_resources(a.resources)).cuda().eval()
    payload=torch.load(a.checkpoint,map_location='cpu')
    if payload['metadata']['config']!=cfg:raise ValueError('Audit checkpoint/config mismatch')
    model.load_learned(payload['ema'])
    if 'runtime_policy' in payload:model.policy=EnginePolicy(**payload['runtime_policy'])
    dataset=build_dataset(mc.dataset.train);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);records=[];states=[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for index,cpu in enumerate(DataLoader(dataset,batch_size=1,num_workers=2,collate_fn=collate)):
            data=to_gpu(cpu);selection,output,_=model.anchor.select(data['inputs'],data['masks'],data['metas'])
            teacher=model.teacher.dense_native(data['inputs'])
            baseline_native,_=model.forward_native(data,selection=selection,apply_router=False)
            feature_error=(baseline_native-teacher).square().mean(1)
            with torch.enable_grad():
                probe=baseline_native.detach().requires_grad_(True);task=model.teacher.loss(probe,data)['cost']
                gradient=torch.autograd.grad(task,probe)[0];absgrad=(gradient*probe).abs().sum(1).detach()
            pairs,targets,rows=paired_interventions(model,data,selection,output,teacher,a.pairs,measure_cost=True)
            if rows:
                predicted=model.router.predict(output,selection,data['masks'],pairs).float().cpu().tolist()
                for row,pred in zip(rows,predicted):
                    r,i=row['remove'],row['insert'];j=row['row']
                    row.update(window=index,predicted_normalized_delta=pred,scales=model.router.scales.cpu().tolist(),
                        actionness_score=float(output['action_logits'][j,i].sigmoid()-output['action_logits'][j,r].sigmoid()),
                        absgrad_score=float(absgrad[j,i//2]-absgrad[j,r//2]),error_score=float(feature_error[j,i//2]-feature_error[j,r//2]),
                        oracle_baselines='absgrad uses training GT; error uses dense external teacher; diagnostic only, never inference inputs')
                records.extend(rows)
            if cfg.get('router'):
                before,_=model.forward_native(data,selection=selection,apply_router=False);after,detail=model.forward_native(data,selection=selection,apply_router=True)
                states.append(dict(window=index,video=data['metas'][0]['video_name'],changes=detail['trace']['routing']['changes'],
                    actual_state_delta=(components(model,before,data)-components(model,after,data)).cpu().tolist()))
            json_write(out/'progress.json',dict(windows=index+1,interventions=len(records)))
            if index+1>=a.windows:break
    with (out/'interventions.jsonl').open('w') as stream:
        for row in records:stream.write(json.dumps(row)+'\n')
    statistics={}
    for i,name in enumerate(('cls','reg')):
        if len(records)>2:
            proxy=np.array([r['repair_delta'][i] for r in records]);actual=np.array([r['actual_delta'][i] for r in records]);correlation=float(spearmanr(proxy,actual)[0])
            statistics[name]=dict(repair_actual_spearman=correlation if np.isfinite(correlation) else None,sign_agreement=float(np.mean(np.sign(proxy)==np.sign(actual))),mean_actual_delta=float(actual.mean()))
            for key in ('actionness_score','absgrad_score','error_score'):
                values=np.array([r[key] for r in records]);value=float(spearmanr(values,actual)[0]) if values.std()>0 and actual.std()>0 else None
                statistics[name][key+'_actual_spearman']=value
    json_write(out/'completed.json',dict(**hardware,config_id=cfg['id'],checkpoint=a.checkpoint,windows=index+1,actual_interventions=len(records),statistics=statistics,
        state_refinements=states,labels='training only; external official teacher repair is distinct from actual RGB computation',not_a_full_test_result=True))

if __name__=='__main__':main()
