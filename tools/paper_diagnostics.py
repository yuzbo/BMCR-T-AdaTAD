"""Save real RGB, temporal provenance, executed masks, and state drift cases."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]
from tools.paper_eval import load_model


def main(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader,Subset
    from opentad.datasets.builder import build_dataset,collate
    from h65.full.runtime import to_gpu
    from h65.paper.runtime import json_write
    from h65.paper.figures import plot_case,save
    import matplotlib.pyplot as plt
    args.need_teacher=True;model,mc,resources,metadata=load_model(args);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    ds=resources['datasets'][model.config['dataset']];annotations=json.loads(Path(ds['annotations']).read_text())['database']
    dataset=build_dataset(mc.dataset.test);first={}
    for index,row in enumerate(dataset.data_list):first.setdefault(row[0],index)
    names=sorted(first);rng=np.random.default_rng(3407);rng.shuffle(names);names=names[:args.videos]
    loader=DataLoader(Subset(dataset,[first[n] for n in names]),batch_size=1,num_workers=2,collate_fn=collate)
    cases=[];summaries=[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for index,cpu in enumerate(loader):
            data=to_gpu(cpu);native,detail=model.forward_native(data,diagnostics=True);trace=detail['trace'];meta=data['metas'][0];name=meta['video_name']
            if model.teacher is not None:target=model.teacher.dense_native(data['inputs']);reference='external official dense teacher'
            else:target,_=model.forward_native(data,force_plan=0,apply_refiner=False);reference='shared full student'
            feature_error=(1-F.cosine_similarity(native.float(),target.float(),dim=1))[0]
            a=detail['selection'].indices.shape[1]//2;h=w=model.encoder.resolution//16
            depth=torch.stack(trace['depth_masks']).reshape(model.encoder.depth,a,h,w).float()
            heavy=torch.stack(trace['spatial_masks']).reshape_as(depth).float()
            block=list(range(1,model.encoder.depth-1,2))[len(range(1,model.encoder.depth-1,2))//2]
            valid=detail['selection'].valid[0].reshape(a,2).any(-1).nonzero().flatten()
            picked=valid[torch.linspace(0,len(valid)-1,3,device=valid.device).round().long()]
            selected=detail['selection'].indices[0]
            rgb=data['inputs'][0,0].index_select(1,selected[(picked*2).long()]).permute(1,2,3,0).detach().float().cpu().numpy()
            if rgb.max()>1:rgb=rgb/255.
            rgb=np.clip(rgb,0,1)
            fps=float(meta.get('fps',-1));duration=float(meta.get('duration',annotations[name].get('duration',0)))
            scale=1/fps if fps>0 else duration/max(1,int(meta['total_frames']))
            times=detail['queries'].frame_times[0].cpu().numpy()*scale;gt=[r['segment'] for r in annotations[name].get('annotations',[]) if r['segment'][1]>=times.min() and r['segment'][0]<=times.max()]
            case=dict(rgb=rgb,overlays=heavy[block,picked].cpu().numpy(),overlay_block=block+1,overlay_indices=picked.cpu().numpy(),
                candidate_times=times,query_times=detail['queries'].centers[0].cpu().numpy()*scale,
                selected_indices=selected.cpu().numpy(),selected_valid=detail['selection'].valid[0].cpu().numpy(),
                contributor_times=detail['anchors'].contributor_times[0].cpu().numpy()*scale,
                actionness=detail['preview']['action_logits'][0].sigmoid().float().cpu().numpy(),gt_segments_seconds=np.asarray(gt).reshape(-1,2),
                depth_fraction=depth.mean((-1,-2)).cpu().numpy(),heavy_fraction=heavy.mean((-1,-2)).cpu().numpy(),
                feature_error=feature_error.cpu().numpy(),plan=detail['plan']['id'],**trace['diagnostic'])
            label=f'{index:02}_{name}';np.savez_compressed(out/f'{label}.npz',**case);plot_case(case,out,label)
            summary=dict(video_name=name,case=label,source_window_index=first[name],reference=reference,plan=detail['plan']['id'],
                seconds_per_source_frame=scale,query_error_mean=float(feature_error.mean()),query_error_p90=float(feature_error.quantile(.9)),
                valid_selected_candidates=int(detail['selection'].valid.sum()),paired_source_gap_seconds=np.diff(case['contributor_times'],axis=1).ravel().tolist(),
                **trace['diagnostic']);summaries.append(summary);cases.append(label)
            json_write(out/'progress.json',dict(cases=cases,total=len(names)));print('Saved actual case '+label,flush=True)
    fig,axes=plt.subplots(1,3,figsize=(12,3.4),layout='constrained')
    for summary in summaries:
        axes[0].plot(summary['layer_ids'],summary['selected_temporal_neighbor_cosine'],alpha=.55,lw=.8)
        axes[1].plot(summary['layer_ids'],summary['spatial_neighbor_cosine'],alpha=.55,lw=.8)
        axes[2].plot(summary['layer_ids'],summary['layer_drift'],alpha=.55,lw=.8)
    for ax,title,label in zip(axes,['Time: selected neighboring states','Space: adjacent patch states','Depth: adjacent block states'],['Cosine similarity','Cosine similarity','1 - cosine similarity']):
        ax.set(xlabel='Block',ylabel=label,title=title)
    fig.suptitle('Activation similarity is a redundancy proxy; task preservation requires the full TAD tests.',fontsize=10)
    save(fig,out,'three_axis_state_distributions');json_write(out/'case_measurements.json',summaries)
    json_write(out/'completed.json',dict(**metadata,cases=cases,actual_traces=True,selection='fixed seed 3407, first window per sampled test video; no score-based case selection',
        gt_usage='drawn only after model execution',compute_scope='diagnostic tensor summaries are excluded from production inference FLOPs'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--resources',default=str(ROOT/'research/paper/resources.local.json'));p.add_argument('--output',required=True)
    p.add_argument('--state',default='ema',choices=['ema','learned']);p.add_argument('--videos',type=int,default=12);main(p.parse_args())
