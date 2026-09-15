"""Full-split official AdaTAD retest with the same complete execution ledger."""
import argparse,copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backbone',choices=['s','b'],required=True);p.add_argument('--output',required=True);a=p.parse_args()
    from h65.full.runtime import initialize_gpu,seed_all,to_gpu
    from h65.paper.runtime import build_config,read_resources,json_write
    from h65.paper.model import PaperModel
    from h65.paper.profile import calibrate_costs
    from h65.paper.evaluation import evaluate
    from h65.paper.geometry import candidate_mask
    from torch.utils.data import DataLoader
    from opentad.datasets.builder import build_dataset,collate
    hardware=initialize_gpu();seed_all(42);resources=read_resources(ROOT/'research/paper/resources.local.json')
    cfg=json.loads((ROOT/f'configs/paper/thumos_{a.backbone}_point_dense_seed42.json').read_text())
    cfg.update(id='public_adatad_'+a.backbone+'_retest',comparison='official_adatad',random_scout=True,initialize_recovery=False,train_head=False,train_adapters=False,train_backbone=False)
    key='thumos:'+a.backbone;source=resources['teachers'][key]
    resources=copy.deepcopy(resources);resources['encoders'][key]=dict(checkpoint=source,kind='task',variant='h65',scout_checkpoint=None)
    mc=build_config(cfg,resources);model=PaperModel(mc,cfg,resources,with_teacher=False).cuda().eval()
    for cpu in DataLoader(build_dataset(mc.dataset.test),batch_size=1,num_workers=2,collate_fn=collate):
        data=to_gpu(cpu)
        if int(candidate_mask(data).sum())==768:break
    else:raise RuntimeError('Official retest has no full window')
    out=Path(a.output);json_write(out/'cost_table.json',calibrate_costs(model,data))
    metadata=dict(hardware,config=cfg,role='external_retested',training_seed=None,evaluation_seed=42,checkpoint=source,checkpoint_state='official_ema',epoch=None,successful_updates=0,source_revision=(ROOT/'source_revision.txt').read_text().strip())
    evaluate(model,mc,resources,out,metadata)
