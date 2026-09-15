"""Executed graph topology, source reliability and same-support state diagnostics."""
import argparse,copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]


def main(args):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from opentad.datasets.builder import build_dataset,collate
    from h65.full.runtime import to_gpu
    from h65.paper.runtime import json_write
    from h65.paper.geometry import candidate_mask
    from h65.paper.support_targets import same_support_targets
    from tools.paper_eval import load_model
    args.need_teacher=True
    model,model_cfg,resources,metadata=load_model(args)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    dataset=build_dataset(model_cfg.dataset.test);cases={};records=[]
    for index,cpu in enumerate(DataLoader(dataset,batch_size=1,num_workers=2,collate_fn=collate)):
        valid=int(candidate_mask(cpu).sum());kind='full' if valid==768 else 'partial' if valid>384 else 'short'
        if kind in cases:continue
        cases[kind]=(index,cpu)
        if len(cases)==3:break
    if 'full' not in cases:raise RuntimeError('No complete diagnostic window')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for kind,(index,cpu) in cases.items():
        data=to_gpu(cpu)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            features,detail=model.forward_native(data,force_plan=4,diagnostics=True,capture_support=True,operator_diagnostics=True)
            terms=same_support_targets(model,data,detail)
            graph_vs_fullkv=None
            if model.config.get('graph_kv'):
                saved=model.config['graph_kv'];model.config['graph_kv']=False
                try:
                    control=dict(model.plan(4));control['route_masks']=dict(depth=detail['trace']['depth_masks'],spatial=detail['trace']['spatial_masks'])
                    plain,_=model.forward_native(data,force_plan=control,selection=detail['selection'],preview=detail['preview'],apply_refiner=False)
                    graph_vs_fullkv=float((features.float()-plain.float()).square().mean()/plain.float().square().mean().clamp_min(1e-6))
                finally:model.config['graph_kv']=saved
        trace=detail['trace'];arrays={};fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        if trace.get('graph_edges'):
            layer=min(trace['graph_edges']);edge=trace['graph_edges'][layer]
            ids=edge['indices'][0].cpu().numpy();weights=edge['weights'][0].float().cpu().numpy()
            actual=trace['query_masks'][layer][0].cpu().numpy()
            incoming=np.zeros(ids.shape[0]);np.add.at(incoming,ids.ravel(),(weights*actual[:,None]).ravel())
            axes[0].imshow(incoming.reshape(8,-1),aspect='auto',origin='lower',cmap='viridis')
            axes[0].set(title=f'Actual edge mass / first packed domain / layer {layer+1}',xlabel='Original spatial patch index',ylabel='Packed tubelet index')
            for l,e in trace['graph_edges'].items():
                for field,value in e.items():arrays[f'layer{l}_{field}']=value.detach().float().cpu().numpy() if value.is_floating_point() else value.detach().cpu().numpy()
        else:axes[0].text(.5,.5,'Encoder uses full KV',ha='center');axes[0].axis('off')
        if trace.get('time_graph_edges'):
            edge=trace['time_graph_edges'];ids=edge['indices'][0].cpu().numpy();weights=edge['weights'][0].float().cpu().numpy()
            valid=detail['queries'].valid[0].cpu().numpy();times=detail['queries'].centers[0].cpu().numpy()/data['metas'][0]['fps']
            quality=edge['geometry'][0,:,3].float().cpu().numpy()
            points=np.flatnonzero(valid);chosen=points[np.linspace(0,len(points)-1,min(24,len(points))).round().astype(int)]
            axes[1].scatter(times[points],quality[points],s=5,c=quality[points],cmap='viridis',vmin=0,vmax=1)
            for i in chosen:
                for slot in np.argsort(weights[i])[-2:]:
                    j=ids[i,slot]
                    if weights[i,slot]>0 and valid[j]:axes[1].plot([times[i],times[j]],[quality[i],quality[j]],alpha=.35,lw=.7,color='#526c8c')
            axes[1].set(title='Original-axis evidence links (24 displayed queries)',xlabel='Source-video time (s)',ylabel='Observed-evidence reliability')
            for field,value in edge.items():arrays['time_'+field]=value.detach().float().cpu().numpy() if value.is_floating_point() else value.detach().cpu().numpy()
        else:axes[1].text(.5,.5,'No temporal graph in G-Context',ha='center');axes[1].axis('off')
        fig.suptitle(f'{model.config["id"]}: {kind} window {index}')
        for ext in ('png','svg'):fig.savefig(out/f'{kind}_graph.{ext}',dpi=180)
        plt.close(fig);np.savez_compressed(out/f'{kind}_edges.npz',**arrays)
        records.append(dict(kind=kind,window_index=index,video_name=data['metas'][0]['video_name'],valid_candidates=int(candidate_mask(data).sum()),
            forced_plan=detail['plan'],source_frame_times=detail['queries'].frame_times,
            same_support_errors=detail['support_diagnostics'],same_support_mean={k:float(v) for k,v in terms.items()},
            same_checkpoint_fixed_masks_graph_vs_full_kv_nmse=graph_vs_fullkv,
            graph_records=trace['graph_records'],graph_router_macs=trace['graph_router_macs'],graph_recovery_macs=trace['graph_recovery_macs'],
            scope='Diagnostic windows only, not full-test mAP; graph/full-KV intervention holds selection and D/S masks fixed'))
    json_write(out/'case_measurements.json',records)
    json_write(out/'completed.json',dict(metadata,cases=len(records),actual_traces=True,graph_diagnostics=True,full_test=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--state',default='ema',choices=['ema','learned'])
    p.add_argument('--resources',default=str(ROOT/'research/paper/resources.local.json'));p.add_argument('--output',required=True)
    main(p.parse_args())
