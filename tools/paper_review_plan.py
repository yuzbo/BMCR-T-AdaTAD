"""Expand the finite review specification and the latest complete-model priorities."""
import argparse,copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.paper_plan import configurations
from h65.paper.runtime import SUPPORT_RECIPE,json_write

def configurations_review(spec):
    old={x['id']:x for x in configurations()};raw={x['id']:x for x in spec['new_runs']};made={}
    def expand(ident):
        if ident in made:return copy.deepcopy(made[ident])
        row=raw[ident];base=row['base'];bb=row['backbone']
        if base in raw:cfg=expand(base)
        elif base in old:cfg=copy.deepcopy(old[base])
        else:cfg=copy.deepcopy(old[f'thumos_{bb}_point_full_seed42'])
        cfg.update(recipe=SUPPORT_RECIPE,id=f'review5485_{ident.lower()}_seed42',review_id=ident,comparison=ident,
                   epochs=row['epochs'],schedule_epochs=row['schedule_epochs'],eval_epochs=row['eval_epochs'],seed=42,backbone=bb)
        changes=row['changes'];mapping={'depth_capacity':'depth_capacity','space_capacity':'space_capacity','kv_mode':'kv_mode','depth_bypass':'depth_bypass','support_loss':'support_loss','capacity_course':'capacity_course','decoder_input_alignment':'decoder_input_alignment'}
        for key,value in changes.items():
            if key not in ('static_keep','source_encoder','head_initialization','decoder_initialization','scout_initialization','actual_queries','frame_label_sampling','frame_context','repair_teacher_queries','decoder_used','shared_full_gt','shared_full_self_feature','external_teacher'):
                cfg[mapping.get(key,key)]=value
        if row['family'] in ('state','static'):
            cfg.update(train_scout=False,frame_utility=False,dynamic_budget=False,train_plan_mode='fixed',fixed_plan=4,
                       support_reference=True,support_diagnostics=True,support_weight=.1,frames=384)
        if row['family']=='static':
            # Latest user decision: simple temporal + static/PBD depth, no dynamic spatial.
            cfg.update(static_compression=True,static_mode='pbd' if changes.get('use_pbd_selection',cfg.get('use_pbd_selection',False)) else 'uniform',
                       depth_capacity=1.,space_capacity=1.,support_layers='retained',train_norm=True,
                       static_target_blocks=9,pbd_round_starts=[0,26,52],candidate_videos=8,
                       epochs=80,schedule_epochs=80,eval_epochs=[10,20,40,60,80])
            cfg['spec_amendment']='Latest P0 complete-course request: 80 epochs and space capacity 1; actual discrete cost match is reported, never assumed exact.'
        if ident=='U00_S':cfg.update(plan_aware_frame=True,frame_label_sampling='stratified_across_plans')
        if ident=='I00_S':cfg.update(instantiate_external_teacher=False,repair_teacher_queries=False)
        if ident in ('I01_S','I02_S'):
            cfg.update(recognition_only=True,instantiate_external_teacher=False,repair_teacher_queries=False,initialize_recovery=False,
                       random_scout=True,random_head=True,decoder='cross',multidepth=True,train_scout=True,train_head=True,train_adapters=True)
            cfg['loss']=dict(task=1.,feature=0.,full_gt=.25,self_feature=.1,action=.1)
        if ident=='I02_S':
            cfg.update(dense_baseline=True,frame_utility=False,dynamic_budget=False,train_scout=False,train_plan_mode='fixed',fixed_plan=0)
            cfg['loss']=dict(task=1.,feature=0.,full_gt=0.,self_feature=0.,action=0.)
        cfg['hypothesis']=row['hypothesis'];cfg['matched_control']=row['matched_control'];cfg['family']=row['family']
        made[ident]=cfg;return copy.deepcopy(cfg)
    rows=[expand(x['id']) for x in spec['new_runs']]
    for bb in ('s','b'):
        cfg=copy.deepcopy(old[f'thumos_{bb}_point_full_seed42']);ident='FULL_V2_'+bb.upper()
        cfg.update(recipe=SUPPORT_RECIPE,id=f'review5485_{ident.lower()}_seed42',review_id=ident,comparison=ident,family='core',
                   mod_start=4,kv_mode='full',depth_bypass='light',support_reference=True,support_loss=True,support_diagnostics=True,support_weight=.1,
                   hypothesis='Complete support-consistent model; first four blocks dense, later alternating updates; temporal selector unchanged.')
        rows.append(cfg)
    early=made['N04_S']
    for ident,uniform in [('N06_S',False),('N07_S',True)]:
        cfg=copy.deepcopy(early);cfg.update(id=f'review5485_{ident.lower()}_seed42',review_id=ident,comparison=ident,mod_start=4,depth_gate='uniform' if uniform else 'attention',matched_control='N04_S' if not uniform else 'N06_S')
        rows.append(cfg)
    cfg=copy.deepcopy(old['thumos_s_point_full_seed42']);cfg.update(recipe=SUPPORT_RECIPE,id='review5485_kd2_s_seed42',review_id='KD2_S',comparison='KD2_S',family='supervision',epochs=40,schedule_epochs=80,eval_epochs=[10,20,40])
    cfg['loss'].update(full_gt=0.,self_feature=0.);cfg['hypothesis']='Remove the entire extra shared-full branch; differs from existing no_self which retains full GT.';rows.append(cfg)
    assert len(rows)==23 and len({x['id'] for x in rows})==23
    return rows

def build(spec):
    rows=configurations_review(spec);stages={}
    core={'FULL_V2_S':2,'FULL_V2_B':3,'P01_S':6,'P01_B':7,'P00_S':8,'P00_B':9}
    for cfg in rows:
        ident=cfg['id'];out=f'research/paper/runs/{ident}';path=f'configs/paper_review/{ident}.json';check='preflight_'+ident
        rank=core.get(cfg['review_id'],20 if cfg['review_id'] in ('N00_S','N01_S','N02_S','N04_S','N06_S','N07_S','R00_S','I00_S','KD2_S') else 40)
        stages[check]=dict(kind='inline_preflight',config_id=ident,runs_with='train_'+ident,done=f'runs/{check}/completed.json',dependencies=[],assets=[])
        stages['train_'+ident]=dict(kind='train',config_id=ident,priority=rank,dependencies=[],assets=[],contains_preflight=check,
            done=f'runs/{ident}/completed.json',resume_checkpoint=f'{out}/latest.pth',args=['tools/paper_course.py','--config',path,'--preflight-output',f'research/paper/runs/{check}','--slice-hours','10'])
        def stage(kind,label,args,requires,priority=25):
            key=f'{kind}_{ident}_{label}';dest=f'{out}/{label}'
            stages[key]=dict(kind=kind,config_id=ident,priority=priority,dependencies=[],assets=[],requires=requires,done=dest.removeprefix('research/paper/')+'/completed.json',args=[*args,'--output',dest])
        stage('eval','terminal_learned',['tools/paper_eval.py','--config',path,'--checkpoint',out+'/terminal.pth','--state','learned'],[out+'/terminal.pth'])
        stage('diagnostics','review_diagnostics',['tools/paper_review_diagnostics.py','--config',path,'--checkpoint',out+'/terminal.pth','--videos','12'],[out+'/terminal.pth'])
        if cfg['review_id'].startswith('FULL_V2') or cfg['review_id']=='U00_S':
            stage('calibrate','calibration',['tools/paper_calibrate.py','--config',path,'--checkpoint',out+'/terminal.pth','--videos','200'],[out+'/terminal.pth'])
            for fraction in (.35,.45,.55,.75,1.02):
                stage('eval',f'budget_{int(fraction*100):03}',['tools/paper_eval.py','--config',path,'--checkpoint',out+'/calibration/calibrated.pth','--budget-fraction',str(fraction)],[out+'/calibration/calibrated.pth'])
        if cfg['review_id'].startswith('FULL_V2'):
            for index in (0,1,2,3,4,12,13,14):
                stage('eval',f'factor_{index:02}_040',['tools/paper_eval.py','--config',path,'--checkpoint',out+'/epoch_040.pth','--force-plan',str(index)],[out+'/epoch_040.pth'])
        if cfg['review_id']=='U00_S':stage('eval','zero_plan_context_040',['tools/paper_eval.py','--config',path,'--checkpoint',out+'/epoch_040.pth','--disable-plan-context'],[out+'/epoch_040.pth'])
    for bb in ('s','b'):
        ident='public_adatad_'+bb+'_retest';dest='research/paper/runs/'+ident
        stages['eval_'+ident]=dict(kind='eval',priority=10,dependencies=[],assets=[],done=f'runs/{ident}/completed.json',
            args=['tools/paper_public_dense.py','--backbone',bb,'--output',dest])
    return dict(recipe=SUPPORT_RECIPE,configs=rows,stages=stages,existing_courses_preserved=52,total_scientific_courses=75,required_seed=42,runs=1,performance_gates=False,
                core_priority=['Full-V1 S','Full-V1 B','Full-V2 S','Full-V2 B','existing Uniform Full S','existing Uniform Full B','PBD-style S','PBD-style B','Static S','Static B'],
                aliases={'C1':'N00_S','C2':'N01_S','C3':'N01_S','C4':'N02_S','C5':'N04_S','C6':'N06_S','C7':'N06_S','C8':'N07_S'},
                note='Eight explanatory labels reuse six distinct courses; Uniform Full preserves its existing dynamic-D/S recipe; T-only control is separate.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',default=str(ROOT/'research/paper/review_5485/EXPERIMENTS.json'));p.add_argument('--emit-only',action='store_true');a=p.parse_args()
    plan=build(json.loads(Path(a.spec).read_text(encoding='utf-8')))
    for cfg in plan['configs']:json_write(ROOT/'configs/paper_review'/f'{cfg["id"]}.json',cfg)
    json_write(ROOT/'research/paper/review_5485/plan.json',plan)
    print(json.dumps(dict(new_configurations=len(plan['configs']),stages=len(plan['stages']),total_courses=75,submitted=False)))
