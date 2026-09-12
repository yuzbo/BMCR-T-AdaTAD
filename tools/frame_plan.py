"""Register parallel routes. Scientific comparison order never gates launching."""
import argparse
import copy
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'research/frame'


def configurations():
    base=dict(temporal_unit='candidate_frame',budget=384,selector='anchor',seed=3407,epochs=20,
        batch_size=1,accumulate=2,ema_decay=.99,learning_rate=1e-4,adapter_lr=1e-5,
        train_execution='dense_mask',train_adapters=False,
        decoder=dict(kind='cross',width=192,layers=2,heads=3,use_provenance=True,use_scout=True),
        engine={},loss=dict(gt_weight=1.,feature_weight=1.),milestones=[5,10,15,20],latency_is_decision_gate=False)
    specs=[]
    def add(name,backbones=('s',),**changes):
        for b in backbones:
            cfg=copy.deepcopy(base);cfg.update(copy.deepcopy(changes));cfg['id']=f'{name}_{b}';cfg['backbone']=b;specs.append(cfg)
    add('R01_interpolate',('s','b'),decoder=dict(kind='interpolate'),evaluate_only=True,epochs=0)
    add('R03_cross',('s','b'))
    add('R02_tcn',('s','b'),decoder=dict(kind='tcn',width=240,layers=2,heads=3,use_provenance=True,use_scout=True))
    add('R04_feature_only',loss=dict(gt_weight=0.,feature_weight=1.))
    add('R05_output_kd',loss=dict(gt_weight=1.,feature_weight=1.,output_kd_weight=.1))
    add('R05_difference',loss=dict(gt_weight=1.,feature_weight=1.,difference_weight=.1))
    add('R07_no_provenance',decoder=dict(kind='cross',width=192,layers=2,heads=3,use_provenance=False,use_scout=True))
    add('R07_no_scout',decoder=dict(kind='cross',width=192,layers=2,heads=3,use_provenance=True,use_scout=False))
    add('D02_amod50',('s','b'),train_adapters=True,engine=dict(depth_schedule='amod',depth_ratio=.5,use_light=False))
    add('D02_amod125',train_adapters=True,engine=dict(depth_schedule='amod',depth_ratio=.125,use_light=False))
    add('D02_amod_uniform',train_adapters=True,engine=dict(depth_schedule='amod',depth_ratio=.5,gate='uniform',use_light=False))
    add('D02_amod_fullkv',train_adapters=True,engine=dict(depth_schedule='amod',depth_ratio=.5,amod_full_kv=True,use_light=False))
    add('D03_amod_compact_train',train_adapters=True,train_execution='compact',engine=dict(depth_schedule='amod',depth_ratio=.5,use_light=False))
    add('D01_static9',train_adapters=True,engine=dict(static_depth=9,use_light=False))
    add('D01_static8',train_adapters=True,engine=dict(static_depth=8,use_light=False))
    add('S01_resolution128',train_adapters=True,resolution=128)
    add('S02_token48',train_adapters=True,engine=dict(spatial_ratio=.48,use_light=True))
    add('S02_tile48',train_adapters=True,engine=dict(spatial_ratio=.48,use_light=True,structured=True))
    add('S03_query50',train_adapters=True,engine=dict(query_ratio=.5,amod_full_kv=True,use_light=False))
    add('U02_actual_utility',router=True,partner_scope='local')
    add('U03_global_partners',router=True,partner_scope='global')
    add('J01_joint',('s','b'),train_adapters=True,engine=dict(depth_schedule='amod',depth_ratio=.5,spatial_ratio=.48,use_light=True),
        train_budgets=[384,768],mixed_gates=True,loss=dict(gt_weight=1.,feature_weight=1.,self_weight=.1))
    add('J02_joint_utility',train_adapters=True,router=True,engine=dict(depth_schedule='amod',depth_ratio=.5,spatial_ratio=.48,use_light=True),
        loss=dict(gt_weight=1.,feature_weight=1.,self_weight=.1))
    add('K01_k320',budget=320);add('K01_k256',budget=256)
    add('G01_seed3408',seed=3408);add('G01_seed3409',seed=3409)
    return specs


def build_plan():
    configs=configurations();stages={}
    for b in ('s','b'):
        stages[f'audit_{b}']=dict(kind='audit',backbone=b,dependencies=[],args=['tools/frame_audit.py','--backbone',b,'--gpu'],done=f'validation/gpu_{b}.json')
    for cfg in configs:
        ident=cfg['id'];path=f'configs/frame/{ident}.json';gate=[f'audit_{cfg["backbone"]}']
        if cfg.get('evaluate_only'):
            stages[f'eval_{ident}']=dict(kind='eval',config_id=ident,dependencies=gate,requires=[],done=f'runs/{ident}_eval_00_ema/completed.json',
                args=['tools/frame_eval.py','--config',path]);continue
        stages[f'train_{ident}']=dict(kind='train',config_id=ident,dependencies=gate,requires=[],done=f'runs/{ident}/completed.json',args=['tools/frame_train.py','--config',path])
        for epoch in cfg['milestones']:
            checkpoint=f'research/frame/runs/{ident}/epoch_{epoch:02}.pth'
            stages[f'eval_{ident}_{epoch:02}']=dict(kind='eval',config_id=ident,epoch=epoch,dependencies=gate,
                requires=[f'runs/{ident}/epoch_{epoch:02}.pth'],done=f'runs/{ident}_eval_{epoch:02}_ema/completed.json',
                args=['tools/frame_eval.py','--config',path,'--checkpoint',checkpoint])
        stages[f'eval_{ident}_online']=dict(kind='eval',config_id=ident,epoch=cfg['epochs'],dependencies=gate,
            requires=[f'runs/{ident}/terminal.pth'],done=f'runs/{ident}_eval_{cfg["epochs"]:02}_learned/completed.json',
            args=['tools/frame_eval.py','--config',path,'--checkpoint',f'research/frame/runs/{ident}/terminal.pth','--state','learned'])
    return dict(recipe='fpw_frame_writeback_amod_v1',source_package='H65_BMCR_3D_CVPR_Agents_Package',
        primary_metrics=['actual_matrix_conv_flops','best_average_mAP'],latency_is_decision_gate=False,
        independent_route_launch=True,configs=configs,stages=stages,max_live_jobs=8,
        external_b01='existing BMCR80 controller, no duplicate warm/joint40',
        conditional_assets=dict(R06='official decoder checkpoint/key-shape audit',G02='second dataset plus matching teacher resources'),
        analysis_dependencies_only='source package R/U/D/S/J ordering is for causal comparisons, not cross-route launch gates')


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiments');p.add_argument('--resources',default=str(EXP/'resources.local.json'));p.add_argument('--dry-run',action='store_true');args=p.parse_args()
    plan=build_plan()
    if args.dry_run:
        print(json.dumps(dict(routes=len(plan['configs']),stages=len(plan['stages']),gpu_execution=False,parallel=True,latency_is_decision_gate=False),indent=2));return
    folder=ROOT/'configs/frame';folder.mkdir(parents=True,exist_ok=True);EXP.mkdir(parents=True,exist_ok=True)
    for cfg in plan['configs']:(folder/f'{cfg["id"]}.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (EXP/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(routes=len(plan['configs']),stages=len(plan['stages']),plan=str(EXP/'plan.json'))))


if __name__=='__main__':main()
