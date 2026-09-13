"""Paper protocol: complete models and independently trained ablations, no mAP gates."""
import argparse
import copy
import itertools
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from h65.paper.runtime import RECIPE,json_write


def configurations():
    base=dict(recipe=RECIPE,dataset='thumos',backbone='s',head='point',seed=3407,epochs=80,eval_epochs=[10,20,40,60,80],
        batch_size=1,accumulate=2,ema_decay=.99,lr=1e-4,head_lr=1e-5,adapter_lr=1e-5,backbone_lr=1e-6,scout_lr=1e-5,
        weight_decay=.05,warmup_epochs=1,selector='anchor',decoder='cross',multidepth=True,provenance=True,scout_context=True,
        train_head=True,train_adapters=True,train_backbone=False,train_scout=True,initialize_recovery=True,
        temporal=True,depth=True,spatial=True,frame_utility=True,dynamic_budget=True,budget_fraction=.55,risk_weight=.25,
        train_plan_mode='mixed',fixed_plan=4,action_interval=8,partner_scope='local',resolution=160,
        loss=dict(task=1.,feature=1.,full_gt=.25,self_feature=.1,action=.1),primary=['actual_complete_model_flops','best_full_test_mAP'])
    rows=[]
    def add(name,backbones=('s',),seeds=(3407,),dataset='thumos',epochs=40,head='point',**changes):
        for backbone,seed in itertools.product(backbones,seeds):
            cfg=copy.deepcopy(base);cfg.update(backbone=backbone,seed=seed,dataset=dataset,epochs=epochs,head=head)
            cfg['eval_epochs']=([5,10,15] if dataset=='anet' else [e for e in (10,20,40,60,80) if e<=epochs])
            if dataset=='anet':cfg.update(head_lr=1e-4,initialize_recovery=False)
            if head=='tadtr':cfg.update(head_lr=1e-4,num_queries=40)
            if backbone=='internvideo_mq':
                cfg['loss'].update(feature=0.,self_feature=1.);cfg.update(initialize_recovery=False,head_lr=1e-4)
            cfg.update(copy.deepcopy(changes));cfg['id']=f'{dataset}_{backbone}_{head}_{name}_seed{seed}';cfg['comparison']=name
            if dataset=='thumos' and backbone in ('s','b') and head=='point' and epochs==40:cfg['schedule_epochs']=80
            rows.append(cfg)
    add('full',('s','b'),(3407,3408,3409),epochs=80)
    add('uniform',('s','b'),(3407,3408,3409),epochs=80,selector='uniform',frame_utility=False)
    dense=dict(dense_baseline=True,train_scout=False,frame_utility=False,dynamic_budget=False,multidepth=False,
               train_plan_mode='fixed',fixed_plan=0,loss=dict(task=1.,feature=0.,full_gt=0.,self_feature=0.,action=0.))
    add('dense',('s','b'),epochs=80,**dense)
    add('random',selector='random',frame_utility=False)
    for name,change in dict(frozen_head=dict(train_head=False),single_level=dict(multidepth=False),
            no_external=dict(loss=dict(task=1.,feature=0.,full_gt=.25,self_feature=.1,action=.1)),
            no_self=dict(loss=dict(task=1.,feature=1.,full_gt=.25,self_feature=0.,action=.1)),
            no_full_gt=dict(loss=dict(task=1.,feature=1.,full_gt=0.,self_feature=.1,action=.1)),
            no_light=dict(use_light=False),no_uncertainty=dict(risk_weight=0.),
            no_action=dict(frame_utility=False,dynamic_budget=False),
            fixed_plan=dict(frame_utility=False,dynamic_budget=False,train_plan_mode='fixed'),
            no_provenance=dict(provenance=False),no_scout_context=dict(scout_context=False),
            frozen_encoder=dict(train_adapters=False,train_scout=False),full_finetune=dict(train_backbone=True),
            tcn=dict(decoder='tcn',multidepth=False),interpolate=dict(decoder='interpolate',multidepth=False),
            mae_pretrained=dict(decoder='mae',mae_pretrained=True,multidepth=False),
            mae_random=dict(decoder='mae',mae_pretrained=False,multidepth=False),
            uniform_attention=dict(attention_uniform=True),full_kv=dict(full_kv=True),tile=dict(structured=True),
            low_resolution=dict(resolution=128),shallow_full=dict(shallow_full=True),global_partners=dict(partner_scope='global')).items():
        add(name,**change)
    for t,d,s in itertools.product((False,True),repeat=3):
        if t and d and s:continue
        add(f'axes_T{int(t)}D{int(d)}S{int(s)}',temporal=t,depth=d,spatial=s,dynamic_budget=False,fixed_plan=4)
    for bb in ('s','b'):
        for name,change in [('full',{}),('uniform',dict(selector='uniform',frame_utility=False)),('dense',dense)]:
            add(name,(bb,),dataset='anet',epochs=15,**change)
    for name,change in [('full',{}),('uniform',dict(selector='uniform',frame_utility=False)),('dense',dense)]:
        add(name,('internvideo_mq',),epochs=40,**change)
        add(name,('s','b'),head='tadtr',epochs=40,**change)
    return rows


def build_plan():
    configs=configurations();stages={}
    # Each native backbone/readout executes two real maximal-model updates first.
    probes={}
    for cfg in configs:
        group=(cfg['dataset'],cfg['backbone'],cfg['head'],'mae' if cfg.get('decoder')=='mae' else 'standard',
               'full_finetune' if cfg.get('train_backbone') else 'adapters')
        if group not in probes:probes[group]=cfg
    def assets(cfg,formal=True):
        result=[]
        if cfg['backbone']=='internvideo_mq':result.append('asset:internvideo_mq')
        elif cfg['dataset']=='anet':result.append('asset:anet:'+cfg['backbone'])
        if cfg.get('decoder')=='mae':result.append('mae:'+cfg['backbone'])
        if cfg['dataset']=='anet' and formal:result.append('data:anet')
        return result
    for group,cfg in probes.items():
        name='preflight_'+cfg['id'];cfg['_preflight']=name
        rank=0 if cfg['dataset']=='thumos' and cfg['backbone']=='s' and cfg['head']=='point' and cfg.get('decoder')=='cross' and not cfg.get('train_backbone') else 1
        stages[name]=dict(kind='preflight',config_id=cfg['id'],priority=rank,dependencies=[],assets=assets(cfg,False),
            done=f'runs/{name}/completed.json',args=['tools/paper_train.py','--config',f'configs/paper/{cfg["id"]}.json','--preflight','--output',f'research/paper/runs/{name}'])
    for cfg in configs:
        group=(cfg['dataset'],cfg['backbone'],cfg['head'],'mae' if cfg.get('decoder')=='mae' else 'standard',
               'full_finetune' if cfg.get('train_backbone') else 'adapters')
        gate=probes[group]['_preflight'];ident=cfg['id'];path=f'configs/paper/{ident}.json';out=f'research/paper/runs/{ident}'
        priority=10 if cfg['comparison']=='full' and cfg['seed']==3407 else 20 if cfg['comparison'] in ('uniform','dense') and cfg['seed']==3407 else 30 if cfg['seed']!=3407 else 40
        stages['train_'+ident]=dict(kind='train',config_id=ident,priority=priority,dependencies=[gate],assets=assets(cfg),
            done=f'runs/{ident}/completed.json',args=['tools/paper_train.py','--config',path,'--resume','--slice-hours','10'])
        def ev(label,epoch,extra=(),state='ema',checkpoint=None,dependency=None,rank=15):
            name=f'eval_{ident}_{label}';checkpoint=checkpoint or f'{out}/epoch_{epoch:03}.pth'
            destination=f'{out}/eval_{label}'
            stages[name]=dict(kind='eval',config_id=ident,epoch=epoch,priority=rank,dependencies=dependency or [gate],assets=assets(cfg),
                requires=[checkpoint],done=f'{destination.removeprefix("research/paper/")}/completed.json',
                args=['tools/paper_eval.py','--config',path,'--checkpoint',checkpoint,'--state',state,'--output',destination,*extra])
        # THUMOS milestone tests execute inside each training allocation.
        if cfg['dataset']=='anet':
            for e in cfg['eval_epochs']:ev(f'{e:03}_ema',e)
        ev('terminal_learned',cfg['epochs'],state='learned',checkpoint=f'{out}/terminal.pth',rank=25)
        if cfg['comparison']=='full' and cfg['seed']==3407:
            terminal=f'{out}/terminal.pth';cal='calibrate_'+ident;cal_out=f'{out}/calibration'
            stages[cal]=dict(kind='calibrate',config_id=ident,priority=16,dependencies=[gate],assets=assets(cfg),requires=[terminal],
                done=f'runs/{ident}/calibration/completed.json',args=['tools/paper_calibrate.py','--config',path,'--checkpoint',terminal,'--output',cal_out,'--videos','200' if cfg['dataset']=='thumos' else '256'])
            for fraction in (.35,.45,.55,.75,1.02):
                ev(f'calibrated_budget{int(fraction*100):03}',cfg['epochs'],['--budget-fraction',str(fraction)],checkpoint=f'{cal_out}/calibrated.pth',dependency=[cal],rank=22)
            diag=f'diagnostics_{ident}'
            stages[diag]=dict(kind='diagnostics',config_id=ident,priority=26,dependencies=[gate],assets=assets(cfg),requires=[terminal],
                done=f'runs/{ident}/diagnostics/completed.json',args=['tools/paper_diagnostics.py','--config',path,'--checkpoint',terminal,'--output',f'{out}/diagnostics','--videos','12'])
            if cfg['dataset']=='thumos' and cfg['head']=='point':
                for idx in (0,1,2,3,4,12,13,14):ev(f'factor{idx:02}_040',40,['--force-plan',str(idx)],rank=23)
                for selector in ('uniform','random'):ev('same_checkpoint_'+selector,40,['--selector',selector,'--disable-frame'],rank=23)
    for cfg in configs:cfg.pop('_preflight',None)
    for cfg in probes.values():
        if cfg['dataset']!='thumos':continue
        ident=cfg['id'];check='preflight_'+ident
        stages[check]['kind']='inline_preflight';stages[check]['runs_with']='train_'+ident
        stage=stages['train_'+ident];stage['dependencies']=[];stage['priority']=stages[check]['priority']
        stage['contains_preflight']=check
        stage['args']=['tools/paper_course.py','--config',f'configs/paper/{ident}.json','--preflight-output',f'research/paper/runs/{check}','--slice-hours','10']
    for cfg in configs:
        if cfg['dataset']!='thumos' or cfg['comparison']!='full' or cfg['seed']!=3407:continue
        ident=cfg['id'];reference=ident.replace('_full_','_uniform_')
        for epoch in ([40,80] if cfg['epochs']==80 else [40]):
            current=f'research/paper/runs/{ident}/eval_{epoch:03}_ema';other=f'research/paper/runs/{reference}/eval_{epoch:03}_ema'
            name=f'paired_{ident}_{epoch:03}'
            stages[name]=dict(kind='analysis',priority=5,dependencies=[],assets=[],
                requires=[f'{current}/completed.json',f'{other}/completed.json'],done=f'runs/{name}/completed.json',
                args=['tools/frame_errors.py','--predictions',f'{current}/result_detection.json','--reference-predictions',f'{other}/result_detection.json',
                      '--output',f'research/paper/runs/{name}','--replicates','1000'])
    return dict(recipe=RECIPE,configs=configs,stages=stages,independent_launch=True,performance_gates=False,
        data_and_technical_dependencies_only=True,legacy_preserved=True,max_live_jobs=10,max_live_train=8,
        comparisons='40-epoch ablations compare with the full model at epoch 40; main THUMOS 80, ANet 15, InternVideo/TadTR 40',
        selection='preregistered checkpoint peak and terminal; no test-driven hyperparameter changes',
        missing_results='null; registered or queued is not executed',latency_is_decision_gate=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dry-run',action='store_true');args=p.parse_args();plan=build_plan()
    if not args.dry_run:
        for cfg in plan['configs']:json_write(ROOT/'configs/paper'/f'{cfg["id"]}.json',cfg)
        json_write(ROOT/'research/paper/plan.json',plan)
    print(json.dumps(dict(configurations=len(plan['configs']),stages=len(plan['stages']),dry_run=args.dry_run)))
