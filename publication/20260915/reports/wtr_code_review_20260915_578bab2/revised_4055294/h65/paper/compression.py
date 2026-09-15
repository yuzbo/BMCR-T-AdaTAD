"""Fixed-budget PBD-style adaptation: train-split loss, original IDs, finite rounds."""
import time
import torch
from opentad.datasets.builder import collate
from h65.full.runtime import to_gpu
from .interventions import evaluation_state
from .profile import execution_flops

def uniform_retained(depth,count):
    values=torch.linspace(0,depth-1,count).round().long().tolist()
    if len(set(values))!=count or values[0]!=0 or values[-1]!=depth-1:raise ValueError('Invalid static layer placement')
    return values

@torch.no_grad()
def advance_compression(model,dataset,epoch):
    if not model.config.get('static_compression'):return None
    cfg=model.config;depth=model.encoder.depth;target=cfg['static_target_blocks']
    current=int(model.compression_stage)
    if cfg['static_mode']=='uniform':
        if current>=0:return None
        keep=uniform_retained(depth,target);model.retained_blocks.zero_();model.retained_blocks[keep]=True;model.compression_stage.fill_(0)
        return dict(mode='static_uniform',epoch=epoch+1,retained_original_ids=keep,candidate_queries=0,candidate_forward_gflops=0.,optimizer_restart=True)
    wanted=sum(epoch>=start for start in cfg['pbd_round_starts'])-1
    if wanted<=current:return None
    if wanted!=current+1:raise RuntimeError('A PBD round cannot be skipped on resume')
    ids=torch.randperm(len(dataset),generator=torch.Generator().manual_seed(cfg['seed']+8100))[:cfg['candidate_videos']].tolist()
    samples=[collate([dataset[i]]) for i in ids]
    before=model.retained_blocks.clone();candidates=before.nonzero().flatten().tolist()
    candidates=[i for i in candidates if i not in (0,depth-1)]
    scores=[];start=time.perf_counter()
    with evaluation_state(model),torch.autocast('cuda',dtype=torch.bfloat16):
        for candidate in candidates:
            model.retained_blocks.copy_(before);model.retained_blocks[candidate]=False
            loss=cls=reg=cost=0.;names=[]
            for cpu in samples:
                data=to_gpu(cpu);features,detail=model.forward_native(data,force_plan=cfg['fixed_plan'],apply_refiner=False)
                values=model.readout.loss(features,data);parts=model.readout.components(values)
                loss+=float(values['cost']);cls+=float(parts[0]);reg+=float(parts[1]);cost+=execution_flops(model,detail)/1e9
                names.append(data['metas'][0]['video_name'])
            scores.append(dict(dropped_original_id=candidate,mean_task_loss=loss/len(samples),mean_cls=cls/len(samples),mean_reg=reg/len(samples),forward_gflops=cost,training_videos=names))
    chosen=min(scores,key=lambda r:(r['mean_task_loss'],r['dropped_original_id']))['dropped_original_id']
    model.retained_blocks.copy_(before);model.retained_blocks[chosen]=False;model.compression_stage.fill_(wanted)
    if int(model.retained_blocks.sum())<target:raise RuntimeError('PBD exceeded the preregistered drop budget')
    return dict(mode='pbd_style_train_loss',epoch=epoch+1,stage=wanted,dropped_original_id=chosen,
                retained_original_ids=model.retained_blocks.nonzero().flatten().tolist(),candidates=scores,
                candidate_queries=len(samples)*len(candidates),candidate_forward_gflops=sum(r['forward_gflops'] for r in scores),
                seconds=time.perf_counter()-start,optimizer_restart=True,course_epochs=cfg['epochs'],
                adaptation='Adapter/norm/head; no LoRA merge; not official faithful reproduction')
