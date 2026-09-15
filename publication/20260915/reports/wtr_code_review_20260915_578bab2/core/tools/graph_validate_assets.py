"""Build every graph variant against actual fixed weights and dataset contracts."""
import json,gc,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]


def main():
    import torch
    from opentad.datasets.builder import build_dataset
    from h65.paper.runtime import build_config,read_config,read_resources,json_write
    from h65.paper.model import PaperModel
    from h65.paper.training import optimizer_groups
    torch.set_num_threads(2)
    resources=read_resources(ROOT/'research/paper/resources.local.json');records=[];matched=None
    for path in sorted((ROOT/'configs/graph').glob('*.json')):
        cfg=read_config(path);torch.manual_seed(cfg['seed']);native=build_config(cfg,resources)
        model=PaperModel(native,cfg,resources,with_teacher=False)
        groups=optimizer_groups(model,cfg)
        parameters={id(p) for p in model.parameters() if p.requires_grad}
        listed=[id(p) for group in groups for p in group['params']]
        if set(listed)!=parameters or len(listed)!=len(parameters):raise RuntimeError('Graph optimizer group omission/duplication')
        learned=model.learned_state();model.load_learned({k:v.clone() for k,v in learned.items()})
        changed_roots=sorted({k.split('.')[0] for k in learned if 'graph' in k})
        protected={k:v.detach().clone() for k,v in model.state_dict().items() if k.startswith(('graph_recovery.','frame_router.network.'))}
        if cfg['id']=='graph_full_s_seed42':matched=protected
        if cfg['id']=='graph_no_referral_s_seed42' and matched is not None:
            if set(protected)!=set(matched) or any(not torch.equal(v,matched[k]) for k,v in protected.items()):
                raise RuntimeError('Shared graph initialization differs in referral control')
        records.append(dict(id=cfg['id'],backbone=cfg['backbone'],epochs=cfg['epochs'],source=model.encoder.provenance,
            parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            learned_keys=len(learned),graph_roots=changed_roots,optimizer_exact_coverage=True,strict_learned_reload=True,
            head_trainable=any(p.requires_grad for p in model.readout.parameters()),source_backbone_trainable=any(p.requires_grad for n,p in model.encoder.vit.named_parameters() if 'adapter' not in n)))
        print(json.dumps(records[-1]),flush=True)
        del model,groups,learned,protected;gc.collect()
    if len(records)!=7:raise RuntimeError('Missing graph configuration')
    train=build_dataset(native.dataset.train);test=build_dataset(native.dataset.test)
    if len(train)!=200 or len(test)!=792:raise RuntimeError('Wrong graph data course size')
    if {r[0] for r in train.data_list}!=set(resources['datasets']['thumos']['train_ids']):raise RuntimeError('Train IDs changed')
    if {r[0] for r in test.data_list}!=set(resources['datasets']['thumos']['test_ids']):raise RuntimeError('Test IDs changed')
    sample=test[2]
    if sample['inputs'].shape!=(1,3,768,160,160) or int(sample['masks'].sum())!=768:raise RuntimeError('Full test tensor contract changed')
    receipt=dict(passed=True,variants=records,train_videos=200,test_videos=211,test_windows=792,
        decoded_shape=list(sample['inputs'].shape),physical_stride=sample['metas']['snippet_stride'],
        gpu_preflight='Not implied by CPU construction; each course has real GPU updates and profiling before training')
    json_write(ROOT/'research/paper/graph/assets_validation.json',receipt)


if __name__=='__main__':main()
