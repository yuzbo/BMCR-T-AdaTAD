#!/usr/bin/env python3
"""Register full courses and scientific dependencies; no experiment results are invented."""
import argparse
import copy
import json
import random
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from h65.paper.runtime import json_write


def main():
    p=argparse.ArgumentParser();p.add_argument('--resources');p.add_argument('--register-only',action='store_true')
    p.add_argument('--site',choices=['4090','a100'],default='4090')
    args=p.parse_args()
    base=json.loads((ROOT/'configs/paper/thumos_s_point_full_seed42.json').read_text())
    specs=[('C0',1.,1.,'uniform','uniform',None),('D-U',.75,1.,'uniform','uniform',None),
        ('D-V',.75,1.,'value','uniform',None),('S-U',1.,.75,'uniform','uniform',None),
        ('S-V',1.,.75,'uniform','value',None),('DS-U',.75,.5,'uniform','uniform','single_axis_value'),
        ('DS-V',.75,.5,'value','value','single_axis_value'),('TDS-V',.75,.5,'value','value','temporal_or_joint_value'),
        ('T-V',1.,1.,'uniform','uniform',None)]
    courses=[]
    for name,d,s,dp,sp,gate in specs:
        cfg=copy.deepcopy(base)
        cfg.update(recipe='wtr_fasttrack_v1',id='wtr_'+name.lower().replace('-','_')+'_s42',comparison=name,
            wtr_fasttrack=True,frames=384,depth_capacity=d,space_capacity=s,selector='uniform',
            train_scout=False,frame_utility=False,dynamic_budget=False,train_plan_mode='fixed',fixed_plan=4,
            instantiate_external_teacher=False,initialize_recovery=False,mod_start=4,kv_mode='full',
            depth_bypass='light',operator_policy={'D':dp,'S':sp},operator_action_interval=8,
            operator_gain_scale=.01,operator_policy_version='wtr_nested_packed_native_v1',
            router_label_protocol='seed42_160_fit_20_calibration_20_holdout',
            primary=['actual_complete_model_flops','epoch80_full_test_mAP'],primary_endpoint_epoch=80,
            requires_evidence=gate,temporal_value=name in ('T-V','TDS-V'),
            loss=dict(task=1.,feature=0.,full_gt=0.,self_feature=.1,action=0.,operator_value=.1))
        if cfg['temporal_value']:cfg.update(selector='anchor',frame_utility=True)
        json_write(ROOT/'configs/wtr_fast'/f'{name}.json',cfg)
        courses.append(dict(id=name,config=f'configs/wtr_fast/{name}.json',
            output=f'research/paper/runs/{cfg["id"]}',status='WAITING_EVIDENCE' if gate else 'REGISTERED',
            requires_evidence=gate,epochs=80,eval_mode='inline',eval_epochs=list(cfg['eval_epochs'])))
    plan=dict(schema='wtr_fasttrack_plan_v3',courses=courses,main_order=[['D-V','D-U'],['S-V','S-U'],['DS-V','DS-U'],['T-V','TDS-V']],
        same_allocation='CPU/dry-run then inline preflight, training, milestone evaluation, checkpoint continuation',
        diagnosis=['A75/F100 direct attention','A100/F75 direct FFN'],
        optional=dict(G0=['Cross_existing','Cross_fresh','StaticGraph','DynamicGraph','DynamicGraph+Referral'],
            G1=['Plain','Graph','Parameter-matched MLP'],RISE=['A actual drift','B same-state function extrapolation'],
            Raw='frozen interface and training-side mini-bank',DB='trained-tier evidence before plan response'),
        final_model_ledger='FINAL_MODEL_LEDGER.csv',publication_tuning=False)
    json_write(ROOT/'research/wtr_fasttrack/plan.json',plan)
    if args.resources:
        import torch
        resources=json.loads(Path(args.resources).read_text())
        payload=torch.load(resources['atlas_light']['s'],map_location='cpu')
        initial=Path(resources['atlas_light']['s']).with_name('v2_s_initial_source.pth')
        if not initial.is_file():raise FileNotFoundError(initial)
        resources['encoders']={'thumos:s':dict(checkpoint=str(initial),scout_checkpoint=str(initial),kind='task',
            variant=payload['metadata']['encoder']['variant'])}
        resources['recovery_initialization']={}
        resources['wtr_initialization']=resources['atlas_light']['s']
        resources['wtr_gate_directory']=str(ROOT/'research/wtr_fasttrack/gates')
        resources['wtr_review_directory']=str(ROOT/'research/wtr_fasttrack/reviews')
        ids=sorted(resources['datasets']['thumos']['train_ids'])
        if len(ids)!=200:raise ValueError('Core router-label split requires the registered 200 training videos')
        random.Random(42).shuffle(ids)
        resources['wtr_router_splits']=dict(fit=sorted(ids[:160]),calibration=sorted(ids[160:180]),holdout=sorted(ids[180:]))
        resources['gpu_type']='A100' if args.site=='a100' else '4090'
        json_write(ROOT/'research/paper/resources.local.json',resources)
        names=['S-V','S-U'] if args.site=='a100' else ['D-V','D-U']
        stages={}
        for name in names:
            course=next(c for c in courses if c['id']==name)
            config=json.loads((ROOT/course['config']).read_text())
            ident='wtr_train_'+config['id']
            stages[ident]=dict(kind='train',priority=0 if name.endswith('-V') else 1,status='WAITING',
                config_id=config['id'],dependencies=[],assets=[],
                requires=[str(ROOT/'research/wtr_fasttrack/reviews'/(config['id']+'.json'))],
                args=[str(ROOT/'tools/paper_course.py'),'--config',str(ROOT/course['config']),
                    '--preflight-output',str(ROOT/'research/paper/runs'/('preflight_'+config['id'])),'--slice-hours','10'],
                done=str(ROOT/'research/paper/runs'/config['id']/'completed.json'),
                resume_checkpoint=str(ROOT/'research/paper/runs'/config['id']/'latest.pth'))
        json_write(ROOT/'research/wtr_fasttrack/deployment_stages.json',stages)
        if args.site=='a100':
            json_write(ROOT/'research/paper/plan.json',dict(recipe='wtr_fasttrack_v1',stages=stages))
            site=ROOT/'research/paper/site';site.mkdir(parents=True,exist_ok=True)
            script='#!/bin/bash\nsource /etc/profile\nset -euo pipefail\ncd '+str(ROOT)+'\nexport OMP_NUM_THREADS=4\nexport OPENBLAS_NUM_THREADS=1\nexec '+sys.executable+' -u "$@"\n'
            (site/'run_job.sh').write_text(script)
    print(json.dumps(dict(registered=len(courses),epochs=80,inline_eval=True,science_sha='not frozen until code validation')))


if __name__=='__main__':main()
