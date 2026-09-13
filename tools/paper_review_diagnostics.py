"""Real RGB, same-support state/operator errors, menu regret, and task-position probes."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]
from tools.paper_eval import load_model

def position_probes(model,data,native,detail):
    import torch
    from h65.paper.geometry import candidate_mask,feature_target_data
    from h65.paper.routing import swap_selection
    masks=candidate_mask(data);t=masks.shape[-1];axis=torch.arange(t,device=masks.device)
    groups={k:torch.zeros(t,dtype=torch.bool,device=masks.device) for k in ('start','interior','end')}
    action_union=torch.zeros(t,dtype=torch.bool,device=masks.device)
    segments=feature_target_data(data)['gt_segments'][0]
    validity=data.get('gt_boundary_validity',[torch.ones_like(segments,dtype=torch.bool)])[0]
    for (a,b),legal in zip(segments,validity):
        action_union|=(axis>=a)&(axis<=b)
        band=max(2.,float(b-a)*.1)
        if bool(legal[0]):groups['start']|=(axis>=a-band)&(axis<=a+band)
        if bool(legal[1]):groups['end']|=(axis>=b-band)&(axis<=b+band)
        groups['interior']|=(axis>a+band)&(axis<b-band)
    groups['end']&=~groups['start'];groups['interior']&=~(groups['start']|groups['end'])
    groups['background']=~(action_union|groups['start']|groups['end'])
    selection=detail['selection'];member=torch.zeros(t,dtype=torch.bool,device=masks.device)
    member[selection.indices[0,selection.valid[0]]]=True;pairs={}
    for name,region in groups.items():
        remove=(region&member&masks[0]).nonzero().flatten().tolist();insert=(~region&~member&masks[0]).nonzero().flatten().tolist();current=[]
        for value in remove:
            if not insert or len(current)==4:break
            chosen=min(insert,key=lambda x:(abs(x-value),x));insert.remove(chosen);current.append((value,chosen))
        pairs[name]=current
    count=min(map(len,pairs.values()))
    if not count:return dict(status='not_all_position_groups_available',common_swaps=0)
    baseline=float(model.readout.loss(native,data)['cost']);rows=[]
    for name,values in pairs.items():
        changed=selection
        for remove,insert in values[:count]:changed=swap_selection(changed,0,remove,insert)
        feature,_=model.forward_native(data,force_plan=detail['plan_index'],selection=changed,preview=detail['preview'],apply_refiner=False)
        loss=float(model.readout.loss(feature,data)['cost'])
        rows.append(dict(group=name,pairs=values[:count],task_loss=loss,loss_increase=loss-baseline))
    return dict(status='complete',common_swaps=count,rows=rows,scope='GT defines offline equal-count frame redistributions only; real re-encoding; diagnostic task loss, not AP',boundary_band='max(2 candidates, 10% GT duration)',group_priority='start, end, interior; background outside all GT and boundary bands; censored endpoints excluded')

def menu_diagnostic(model,data,detail):
    import torch
    from h65.paper.profile import execution_flops
    if not model.config.get('dynamic_budget'):return None
    rows=[]
    for index in range(len(model.menu)):
        native,current=model.forward_native(data,force_plan=index,preview=detail['preview'],apply_refiner=False)
        losses=model.readout.loss(native,data);parts=model.readout.components(losses)
        rows.append(dict(plan_index=index,plan=current['plan']['id'],task_loss=float(losses['cost']),components=parts.cpu().tolist(),gflops=execution_flops(model,current)/1e9))
    feasible=(model.budget_router.cost_gflops<=model.budget_router.reference_gflops*model.config['budget_fraction']).nonzero().flatten().tolist()
    chosen=detail['plan_index'];best=min(feasible,key=lambda i:rows[i]['task_loss'])
    mu,lv=model.budget_router.distribution(detail['context'])
    return dict(rows=rows,chosen=chosen,oracle=best,task_loss_regret=rows[chosen]['task_loss']-rows[best]['task_loss'],
                feasible=feasible,predicted_mean=(mu[0]*model.budget_router.scales).cpu().tolist(),predicted_sigma=(lv[0].mul(.5).exp()*model.budget_router.scales).cpu().tolist(),
                scope='fixed student, unrefined per-plan policy matching action labels; GT is diagnostic only; loss regret is not mAP regret')

def main(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader,Subset
    from opentad.datasets.builder import build_dataset,collate
    from h65.full.runtime import to_gpu
    from h65.paper.runtime import json_write
    from h65.paper.support_targets import same_support_targets,compare_states
    from h65.paper.figures import save,plot_case
    import matplotlib.pyplot as plt
    args.need_teacher=True;model,mc,resources,metadata=load_model(args);model.ensure_support_reference()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);dataset=build_dataset(mc.dataset.test);first={}
    annotations=json.loads(Path(resources['datasets'][model.config['dataset']]['annotations']).read_text())['database']
    for index,row in enumerate(dataset.data_list):first.setdefault(row[0],index)
    names=sorted(first);np.random.default_rng(model.config['seed']).shuffle(names);names=names[:args.videos]
    loader=DataLoader(Subset(dataset,[first[n] for n in names]),batch_size=1,num_workers=2,collate_fn=collate)
    summaries=[];cases=[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for index,cpu in enumerate(loader):
            data=to_gpu(cpu);native,detail=model.forward_native(data,diagnostics=True,capture_support=True,operator_diagnostics=True)
            same_support_targets(model,data,detail);trace=detail['trace'];selection=detail['selection'];meta=data['metas'][0];name=meta['video_name']
            k=selection.indices.shape[1];a=k//2;h=w=model.encoder.resolution//16
            _,_,current_full_trace=model.encoder.encode(data['inputs'],selection,dict(frames=k,depth=1.,space=1.),capture='diagnostic',state_capture=True)
            valid=selection.valid.reshape(1,a,2).any(-1).reshape(k//16,8).repeat_interleave(h*w,1)
            _,current_errors=compare_states(trace['state_taps'],current_full_trace['state_taps'],valid)
            del current_full_trace
            if model.teacher is not None:target=model.teacher.dense_native(data['inputs']);reference='external original-axis teacher'
            else:target,_=model.forward_native(data,force_plan=0,apply_refiner=False);reference='shared full student'
            error=(1-F.cosine_similarity(native.float(),target.float(),dim=1))[0].masked_fill(~detail['queries'].valid[0],float('nan'))
            depth=torch.stack(trace['depth_masks']).reshape(model.encoder.depth,a,h,w).float();heavy=torch.stack(trace['spatial_masks']).reshape_as(depth).float()
            active=detail['plan'].get('mod_layers',list(range(1,model.encoder.depth-1,2)));block=active[len(active)//2]
            choices=selection.valid[0].reshape(a,2).any(-1).nonzero().flatten();picked=choices[torch.linspace(0,len(choices)-1,3,device=choices.device).round().long()]
            selected=selection.indices[0];rgb=data['inputs'][0,0].index_select(1,selected[picked*2]).permute(1,2,3,0).float().cpu().numpy()
            if rgb.max()>1:rgb=rgb/255.
            fps=float(meta.get('fps',-1));duration=float(meta.get('duration',annotations[name].get('duration',0)));scale=1/fps if fps>0 else duration/max(1,int(meta['total_frames']))
            times=detail['queries'].frame_times[0].cpu().numpy()*scale;gt=[r['segment'] for r in annotations[name].get('annotations',[]) if r['segment'][1]>=times.min() and r['segment'][0]<=times.max()]
            case=dict(rgb=np.clip(rgb,0,1),overlays=heavy[block,picked].cpu().numpy(),overlay_block=block+1,overlay_indices=picked.cpu().numpy(),
                candidate_times=times,candidate_valid=detail['queries'].candidate_mask[0].cpu().numpy(),query_times=detail['queries'].centers[0].cpu().numpy()*scale,
                selected_indices=selected.cpu().numpy(),selected_valid=selection.valid[0].cpu().numpy(),contributor_times=detail['anchors'].contributor_times[0].cpu().numpy()*scale,
                actionness=detail['preview']['action_logits'][0].sigmoid().float().cpu().numpy(),gt_segments_seconds=np.asarray(gt).reshape(-1,2),
                depth_fraction=depth.mean((-1,-2)).cpu().numpy(),heavy_fraction=heavy.mean((-1,-2)).cpu().numpy(),feature_error=error.cpu().numpy(),plan=detail['plan']['id'],**trace['diagnostic'])
            label=f'{index:02}_{name}';np.savez_compressed(out/f'{label}.npz',**case);plot_case(case,out,label)
            exposures={key:trace[key] for key in ('query_masks','kv_masks','ffn_light_masks','depth_bypass_masks','age_before_reentry','q','kv','heavy_mlp','light','depth_attention_light','depth_ffn_light','tia','score_qk')}
            json_write(out/f'{label}_exposure.json',exposures)
            row=dict(video_name=name,source_window_index=first[name],reference=reference,plan=detail['plan'],support_key=detail['support_key'],
                     fixed_initial_support_errors=detail['support_diagnostics'],same_checkpoint_support_errors=current_errors,operator_errors=trace['operator_errors'],
                     original_axis_error=float(error[detail['queries'].valid[0]].mean()),case=label)
            trace.pop('state_taps',None)
            row['position_probes']=position_probes(model,data,native,detail)
            row['menu_diagnostic']=menu_diagnostic(model,data,detail)
            json_write(out/f'{label}_measurements.json',row);summaries.append(row);cases.append(label)
            json_write(out/'progress.json',dict(cases=cases,total=len(names)));print('Completed support/state case '+label,flush=True)
    fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
    for ax,point in zip(axes,('attention','pre_tia','post_tia')):
        for row in summaries:
            items=[x for x in row['same_checkpoint_support_errors'] if x['point']==point]
            ax.plot([x['original_block_id'] for x in items],[float(x['normalized_mse']) for x in items],alpha=.5,lw=.8)
        ax.set(xlabel='Original block ID (0 based)',ylabel='Normalized state MSE',title=point)
    save(fig,out,'same_support_layer_errors')
    json_write(out/'case_measurements.json',summaries)
    json_write(out/'completed.json',dict(metadata,cases=cases,actual_traces=True,same_support_reference=True,
        selection='seed42 fixed sample, first test window per selected video',gt_usage='offline diagnosis only',scope='diagnostic overhead excluded from production profiles'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--resources',default=str(ROOT/'research/paper/resources.local.json'));p.add_argument('--state',default='ema',choices=['ema','learned']);p.add_argument('--videos',type=int,default=12);p.add_argument('--output',required=True);main(p.parse_args())
