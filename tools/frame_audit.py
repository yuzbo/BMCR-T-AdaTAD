"""Real legacy/native, zero-decoder, A-MoD packing, and task-gradient checks."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]
from h65.frame.runtime import EXP,json_write


def main():
    p=argparse.ArgumentParser();p.add_argument('--backbone',choices=['s','b'],default='s');p.add_argument('--resources',default=str(EXP/'resources.local.json'))
    p.add_argument('--base-commit',default='239d098cd899936c35989fae6243c70259a85adb');p.add_argument('--output');p.add_argument('--gpu',action='store_true');p.add_argument('--dry-run',action='store_true');args=p.parse_args()
    if args.dry_run or not args.gpu:
        print(json.dumps(dict(backbone=args.backbone,resources=args.resources,gpu_execution=False,checks=['legacy_native','zero_residual','amod_packing','student_GT_gradient'])));return
    import torch
    from dataclasses import replace
    from torch.utils.data import DataLoader
    from opentad.datasets.builder import build_dataset,collate
    from h65.full.runtime import initialize_gpu,to_gpu
    from h65.frame.runtime import data_config,read_resources,read_config
    from h65.frame.model import FrameModel
    from h65.frame.geometry import interpolate_anchors,make_anchors,make_queries,scout_context
    from h65.frame.contracts import EnginePolicy
    from h65.frame.objectives import training_objectives
    from tools.full_eval import profile_case
    hardware=initialize_gpu();cfg=read_config(ROOT/f'configs/frame/R03_cross_{args.backbone}.json');cfg['train_adapters']=True
    config=data_config(args.backbone);resources=read_resources(args.resources);model=FrameModel(config.model,cfg,resources).cuda().eval()
    dataset=build_dataset(config.dataset.test);samples={}
    for i,data in enumerate(DataLoader(dataset,batch_size=1,num_workers=2,collate_fn=collate)):
        samples.setdefault(profile_case(data),data)
        if len(samples)==3:break
    identity={};engine_checks={}
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for kind,cpu in samples.items():
            data=to_gpu(cpu);selection,output,_=model.anchor.select(data['inputs'],data['masks'],data['metas'])
            old,_=model.anchor.model.encode(data['inputs'],selection)
            native,trace=model.anchor.encode_native(data['inputs'],selection)
            transform=model.anchor.model.backbone.post_processing_pipeline.transforms[-1]
            recovered=transform(dict(feats=native.transpose(1,2)))['feats'].float()*selection.valid[:,None]
            error=float((old-recovered).abs().max())
            if not torch.allclose(old,recovered,atol=2e-4,rtol=2e-4):raise RuntimeError(f'Legacy native mismatch:{kind}:{error}')
            anchors=make_anchors(native,selection,data['masks'],data['metas'],trace);queries=make_queries(data['masks'],data['metas'],selection)
            base=interpolate_anchors(anchors,queries).transpose(1,2)
            decoded=model.decoder(anchors,queries,scout_context(output,data['masks']))
            if not torch.equal(decoded,base):raise RuntimeError('Zero residual does not equal interpolation')
            identity[kind]=dict(max_abs_error=error,zero_residual_equal=True,valid_candidates=int(data['masks'].sum()))
        data=to_gpu(samples['full']);selection,_,_=model.anchor.select(data['inputs'],data['masks'],data['metas'])
        clips=model.anchor.prepare_clips(model.anchor.selected_rgb(data['inputs'],selection));valid=selection.valid.reshape(1,-1,2).any(-1)
        full,h,w,trace=model.engine(model.anchor.vit,clips,EnginePolicy(),valid)
        reference=model.anchor.vit(clips)
        actual=model.anchor.vit.norm(full).reshape(len(clips),8,h,w,model.anchor.vit.embed_dims).permute(0,4,1,2,3)
        if not torch.allclose(actual,reference,atol=2e-4,rtol=2e-4):raise RuntimeError('Full gate differs from parent ViT')
        for name,policy in [('amod50',EnginePolicy(depth_schedule='amod',depth_ratio=.5,use_light=False)),
                            ('joint',EnginePolicy(depth_schedule='amod',depth_ratio=.5,spatial_ratio=.48,use_light=True)),
                            ('query',EnginePolicy(query_ratio=.5,amod_full_kv=True,use_light=False))]:
            compact,_,_,trace=model.engine(model.anchor.vit,clips,policy,valid)
            masks=dict(depth={i:x for i,x in enumerate(trace['depth_masks'])},spatial={i:x for i,x in enumerate(trace['spatial_masks'])},query={i:x for i,x in enumerate(trace['query_masks'])})
            paired=replace(policy,mode='dense_mask',route_masks=masks)
            dense,_,_,other=model.engine(model.anchor.vit,clips,paired,valid)
            error=float((compact-dense).abs().max());rms=float((compact-dense).float().square().mean().sqrt())
            if not torch.allclose(compact,dense,atol=.04,rtol=.015):raise RuntimeError(f'AMP packed/masked mismatch:{name}:{error}/{rms}')
            if not (trace['q'][0]==trace['q'][-1]==len(clips)*800):raise RuntimeError('First/last layer not dense')
            engine_checks[name]=dict(max_abs_error=error,rms_error=rms,actual_q=trace['q'],actual_kv=trace['kv'],heavy_mlp=trace['heavy_mlp'],score_qk_macs=trace['score_qk'])
    train=build_dataset(config.dataset.train);loader=DataLoader(train,batch_size=1,num_workers=2,collate_fn=collate)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    model.train();normalizer=model.teacher.model.detector.rpn_head.loss_normalizer.detach().clone();gradient_norms=[]
    for step,cpu in enumerate(loader):
        data=to_gpu(cpu);optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            target=model.teacher.dense_native(data['inputs']);native,_=model.forward_native(data)
            losses=training_objectives(native,target,data,model.teacher)
            gradient=torch.autograd.grad(losses['gt_loss'],native,retain_graph=True)[0]
        if gradient.abs().sum()==0 or not torch.isfinite(gradient).all():raise RuntimeError('GT gradient missing')
        losses['cost'].backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        optimizer.step();gradient_norms.append(float(gradient.abs().mean()))
        if step==1:break
    if not torch.equal(normalizer,model.teacher.model.detector.rpn_head.loss_normalizer):raise RuntimeError('Frozen detector normalizer changed')
    if any(p.grad is not None for p in model.teacher.parameters()):raise RuntimeError('Teacher received parameter gradients')
    result=dict(**hardware,backbone=args.backbone,weights_loaded=True,legacy_identity=identity,engine=engine_checks,
                real_task_updates=2,student_gt_gradient=gradient_norms,teacher_frozen=True,normalizer_unchanged=True,
                anchor=model.anchor.provenance,project_GPU_checks=True,not_a_full_test_result=True)
    out=Path(args.output) if args.output else EXP/f'validation/gpu_{args.backbone}.json';json_write(out,result);print(json.dumps(result))


if __name__=='__main__':main()
