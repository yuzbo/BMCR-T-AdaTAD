"""Register parallel routes. Scientific comparison order never gates launching."""
import argparse
import copy
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'research/frame'


def configurations(decoder_assets=None):
    base=dict(temporal_unit='candidate_frame',budget=384,selector='anchor',seed=3407,epochs=20,
        batch_size=1,accumulate=2,ema_decay=.99,learning_rate=1e-4,adapter_lr=1e-5,
        train_execution='compact',train_adapters=False,
        decoder=dict(kind='cross',width=192,layers=2,heads=3,use_provenance=True,use_scout=True),
        engine={},loss=dict(gt_weight=1.,feature_weight=1.),milestones=[5,10,15,20],latency_is_decision_gate=False)
    specs=[]
    def add(name,backbones=('s','b'),**changes):
        for b in backbones:
            cfg=copy.deepcopy(base);cfg.update(copy.deepcopy(changes));cfg['id']=f'{name}_{b}';cfg['backbone']=b;specs.append(cfg)
    add('R01_interpolate',('s','b'),decoder=dict(kind='interpolate'),evaluate_only=True,epochs=0)
    add('R01_anchor_head',('s','b'),decoder=dict(kind='interpolate'),evaluate_only=True,epochs=0,detector_source='anchor')
    add('C01_vector_condition',('b',),decoder=dict(kind='interpolate'),evaluate_only=True,epochs=0,vector_condition=True)
    add('R03_cross',('s','b'))
    add('R02_tcn',('s','b'),decoder=dict(kind='tcn',width=240,layers=2,heads=3,use_provenance=True,use_scout=True))
    add('R04_feature_only',loss=dict(gt_weight=0.,feature_weight=1.))
    add('R05_output_kd',loss=dict(gt_weight=1.,feature_weight=1.,output_kd_weight=.1))
    add('R05_difference',loss=dict(gt_weight=1.,feature_weight=1.,difference_weight=.1))
    add('R07_no_provenance',decoder=dict(kind='cross',width=192,layers=2,heads=3,use_provenance=False,use_scout=True))
    add('R07_no_scout',decoder=dict(kind='cross',width=192,layers=2,heads=3,use_provenance=True,use_scout=False))
    add('D00_dense_adapters',train_adapters=True)
    add('D02_amod50',('s','b'),train_adapters=True,engine=dict(depth_schedule='amod',depth_ratio=.5,use_light=False))
    add('D02_amod125',train_adapters=True,engine=dict(depth_schedule='amod',depth_ratio=.125,use_light=False))
    add('D02_amod_uniform',train_adapters=True,engine=dict(depth_schedule='amod',depth_ratio=.5,gate='uniform',use_light=False))
    add('D02_amod_fullkv',train_adapters=True,engine=dict(depth_schedule='amod',depth_ratio=.5,amod_full_kv=True,use_light=False))
    add('D03_amod_dense_mask_train',train_adapters=True,train_execution='dense_mask',engine=dict(depth_schedule='amod',depth_ratio=.5,use_light=False))
    add('D01_static9',train_adapters=True,engine=dict(static_depth=9,use_light=False))
    add('D01_static8',train_adapters=True,engine=dict(static_depth=8,use_light=False))
    add('D01_pbd8',train_adapters=True,engine=dict(static_depth=12,use_light=False),progressive_drop_updates=[200,400,600,800])
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
    for budget in (320,256):
        add(f'K01_joint{budget}',budget=budget,train_adapters=True,train_budgets=[budget,384],mixed_gates=True,
            engine=dict(depth_schedule='amod',depth_ratio=.5,spatial_ratio=.48,use_light=True),loss=dict(gt_weight=1.,feature_weight=1.,self_weight=.1))
    add('P01_full_stem80',('s','b'),shallow_full_resolution=80)
    add('G01_seed3408',seed=3408);add('G01_seed3409',seed=3409)
    for b,asset in (decoder_assets or {}).items():
        decoder=dict(kind='mae',**{k:asset[k] for k in ('width','layers','heads')})
        add('R06_mae_random',(b,),decoder=decoder,decoder_initialization='random')
        add('R06_mae_pretrained',(b,),decoder=decoder,decoder_initialization='videomae')
    return specs


def build_plan(decoder_assets=None):
    configs=configurations(decoder_assets);stages={}
    for b in ('s','b'):
        stages[f'audit_{b}']=dict(kind='audit',backbone=b,dependencies=[],args=['tools/frame_audit.py','--backbone',b,'--gpu','--scope','all'],done=f'validation/gpu_{b}.json')
        stages[f'audit_recovery_{b}']=dict(kind='audit',backbone=b,dependencies=[],args=['tools/frame_audit.py','--backbone',b,'--gpu','--scope','recovery','--output',f'research/frame/validation/gpu_recovery_{b}.json'],done=f'validation/gpu_recovery_{b}.json')
    for cfg in configs:
        ident=cfg['id'];path=f'configs/frame/{ident}.json'
        gate=[f'audit_{cfg["backbone"]}'] if cfg.get('train_adapters') or cfg.get('engine') else [f'audit_recovery_{cfg["backbone"]}']
        if cfg['decoder']['kind']=='mae' or cfg.get('shallow_full_resolution') or ident=='J01_joint_b':
            checked=f'R06_mae_pretrained_{cfg["backbone"]}' if cfg['decoder']['kind']=='mae' else ident
            check=f'preflight_{checked}'
            stages.setdefault(check,dict(kind='module_preflight',config_id=checked,epoch=0,dependencies=gate.copy(),
                done=f'runs/{check}/completed.json',args=['tools/frame_train.py','--config',f'configs/frame/{checked}.json','--preflight']))
            gate=gate+[check]
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
        name=f'A01_analysis_{ident}_20';baseline=f'R01_interpolate_{cfg["backbone"]}'
        stages[name]=dict(kind='analysis',config_id=ident,dependencies=[f'eval_{ident}_20',f'eval_{baseline}'],
            done=f'runs/{name}/completed.json',args=['tools/frame_errors.py','--predictions',f'research/frame/runs/{ident}_eval_20_ema/result_detection.json',
                '--reference-predictions',f'research/frame/runs/{baseline}_eval_00_ema/result_detection.json','--output',f'research/frame/runs/{name}','--replicates','200'])
    for ident in (f'{route}_{b}' for route in ('R03_cross','U02_actual_utility','J01_joint') for b in ('s','b')):
        stage=f'intervene_{ident}_05';checkpoint=f'research/frame/runs/{ident}/epoch_05.pth'
        stages[stage]=dict(kind='intervene',config_id=ident,epoch=5,dependencies=stages[f'train_{ident}']['dependencies'],
            requires=[f'runs/{ident}/epoch_05.pth'],done=f'runs/{stage}/completed.json',
            args=['tools/frame_intervene.py','--config',f'configs/frame/{ident}.json','--checkpoint',checkpoint,'--windows','32','--pairs','2','--output',f'research/frame/runs/{stage}'])
    for b in ('s','b'):
        ident=f'R03_cross_{b}'
        for selector in ('uniform','random'):
            name=f'U04_{ident}_{selector}_05';out=f'research/frame/runs/{name}'
            stages[name]=dict(kind='eval',config_id=ident,epoch=5,dependencies=[f'audit_recovery_{b}'],requires=[f'runs/{ident}/epoch_05.pth'],done=f'runs/{name}/completed.json',
                args=['tools/frame_eval.py','--config',f'configs/frame/{ident}.json','--checkpoint',f'research/frame/runs/{ident}/epoch_05.pth','--selector',selector,'--output',out])
        ident=f'J01_joint_{b}'
        for epoch in (5,20):
            for budget in (384,768):
                for depth in (1.,.5):
                    for spatial in (1.,.48):
                        if (budget,depth,spatial)==(384,.5,.48):continue # already registered EMA primary evaluation
                        name=f'J01_factor_{b}_K{budget}_D{int(100*depth)}_S{int(100*spatial)}_{epoch:02}'
                        stages[name]=dict(kind='eval',config_id=ident,epoch=epoch,dependencies=[f'audit_{b}'],requires=[f'runs/{ident}/epoch_{epoch:02}.pth'],done=f'runs/{name}/completed.json',
                            args=['tools/frame_eval.py','--config',f'configs/frame/{ident}.json','--checkpoint',f'research/frame/runs/{ident}/epoch_{epoch:02}.pth',
                                '--budget',str(budget),'--policy-json',json.dumps(dict(depth_ratio=depth,spatial_ratio=spatial)),'--output',f'research/frame/runs/{name}'])
        name=f'M01_interleaved_{b}_20'
        stages[name]=dict(kind='benchmark',backbone=b,dependencies=[f'audit_{b}'],
            requires=[f'runs/{route}_{b}/epoch_20.pth' for route in ('R03_cross','D02_amod50','J01_joint')],done=f'runs/{name}/completed.json',
            args=['tools/frame_benchmark.py','--backbone',b,'--epoch','20','--output',f'research/frame/runs/{name}'])
    for b in ('s','b'):
        name=f'U04_router_off_{b}_20';ident=f'U02_actual_utility_{b}'
        stages[name]=dict(kind='eval',config_id=ident,epoch=20,dependencies=[f'audit_recovery_{b}'],requires=[f'runs/{ident}/epoch_20.pth'],done=f'runs/{name}/completed.json',
            args=['tools/frame_eval.py','--config',f'configs/frame/{ident}.json','--checkpoint',f'research/frame/runs/{ident}/epoch_20.pth','--disable-router','--output',f'research/frame/runs/{name}'])
    return dict(recipe='fpw_frame_writeback_amod_v1',source_package='H65_BMCR_3D_CVPR_Agents_Package',
        primary_metrics=['actual_matrix_conv_flops','best_average_mAP'],latency_is_decision_gate=False,
        independent_route_launch=True,configs=configs,stages=stages,max_live_jobs=8,
        external_b01='existing BMCR80 controller, no duplicate warm/joint40',
        conditional_assets=dict(R06_missing_backbones=[b for b in ('s','b') if b not in (decoder_assets or {})],G02='full second dataset plus matching task-adapted teacher resources'),
        max_cpu_analysis=1,
        analysis_dependencies_only='source package R/U/D/S/J ordering is for causal comparisons, not cross-route launch gates')


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiments');p.add_argument('--resources',default=str(EXP/'resources.local.json'));p.add_argument('--dry-run',action='store_true');args=p.parse_args()
    resources=Path(args.resources)
    assets=json.loads(resources.read_text()).get('decoder_pretrain',{}) if resources.exists() else {}
    plan=build_plan(assets)
    if args.dry_run:
        print(json.dumps(dict(routes=len(plan['configs']),stages=len(plan['stages']),gpu_execution=False,parallel=True,latency_is_decision_gate=False),indent=2));return
    folder=ROOT/'configs/frame';folder.mkdir(parents=True,exist_ok=True);EXP.mkdir(parents=True,exist_ok=True)
    for cfg in plan['configs']:(folder/f'{cfg["id"]}.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (EXP/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(routes=len(plan['configs']),stages=len(plan['stages']),plan=str(EXP/'plan.json'))))


if __name__=='__main__':main()
