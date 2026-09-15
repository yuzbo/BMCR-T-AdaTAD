"""Four complete graph candidates and three independent S mechanism controls."""
import argparse,copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from h65.paper.runtime import GRAPH_RECIPE,json_write


def configurations():
    rows=[]
    specs=[('repair','s',80,False,True,False,'referral'),('context','s',80,True,False,False,'referral'),
           ('full','s',80,True,True,True,'referral'),('full','b',80,True,True,True,'referral'),
           ('fixed_local','s',40,True,True,True,'fixed_local'),('no_referral','s',40,True,True,True,'no_referral'),
           ('full_kv','s',40,False,True,True,'referral')]
    for name,bb,epochs,kv,recovery,coupled,mode in specs:
        cfg=json.loads((ROOT/f'configs/paper_review/review5485_full_v2_{bb}_seed42.json').read_text(encoding='utf-8'))
        cfg.update(recipe=GRAPH_RECIPE,id=f'graph_{name}_{bb}_seed42',comparison='G-'+name.replace('_','-'),
            family='graph_core' if epochs==80 else 'graph_mechanism',epochs=epochs,schedule_epochs=80,
            eval_epochs=[10,20,40,60,80] if epochs==80 else [10,20,40],seed=42,
            graph_kv=kv,graph_recovery=recovery,graph_couplings=[6,9,12] if coupled else [],
            graph_frame=coupled,graph_mode=mode,graph_referrals=2 if mode=='referral' else 0,
            graph_transition_epochs=5,coverage_bins=32 if coupled else 0,
            anchor_consistency_weight=.02 if recovery else 0.,plan_aware_frame=coupled)
        cfg.pop('review_id',None)
        cfg['hypothesis']='Sparse relation access and contributor-aware original-axis repair; compare complete measured cost and full-test accuracy.'
        cfg['initialization_contract']='Same frozen task/recognition and R03 assets as Full-V2; new graph module-family RNGs are fixed seed42 offsets, not extra experimental seeds.'
        rows.append(cfg)
    return rows


def build(root):
    root=Path(root);rows=configurations();stages={}
    priority={'full_s':-2,'full_b':-2,'repair_s':0,'context_s':0,'fixed_local_s':4,'no_referral_s':4,'full_kv_s':4}
    for cfg in rows:
        ident=cfg['id'];short=ident.removeprefix('graph_').removesuffix('_seed42')
        out=root/'research/paper/runs'/ident;config=root/'configs/graph'/f'{ident}.json';audit=root/'research/paper/runs'/('preflight_'+ident)
        train='train_'+ident;preflight='preflight_'+ident;standalone=short in ('full_s','full_b')
        stages[preflight]=dict(kind='preflight' if standalone else 'inline_preflight',config_id=ident,priority=-4,
            runs_with=train,done=str(audit/'completed.json'),dependencies=[],assets=[],
            args=[str(root/'tools/paper_train.py'),'--config',str(config),'--preflight','--output',str(audit)])
        stages[train]=dict(kind='train',config_id=ident,priority=priority[short],dependencies=[],assets=[],
            contains_preflight=preflight,done=str(out/'completed.json'),resume_checkpoint=str(out/'latest.pth'),
            requires=[str(audit/'completed.json')] if standalone else [],
            args=[str(root/'tools/paper_course.py'),'--config',str(config),'--preflight-output',str(audit),'--slice-hours','10'])
        def stage(kind,label,tool,extra,requires,rank=10):
            stages[kind+'_'+ident+'_'+label]=dict(kind=kind,config_id=ident,priority=rank,dependencies=[],assets=[],requires=requires,
                done=str(out/label/'completed.json'),args=[str(root/'tools'/tool),'--config',str(config),*extra,'--output',str(out/label)])
        stage('eval','terminal_learned','paper_eval.py',['--checkpoint',str(out/'terminal.pth'),'--state','learned'],[str(out/'terminal.pth')])
        stage('graph_diagnostics','graph_diagnostics','graph_diagnostics.py',['--checkpoint',str(out/'epoch_040.pth'),'--state','ema'],[str(out/'epoch_040.pth')],rank=3)
        if short in ('full_s','full_b'):
            for index in (1,4,9,10):
                stage('eval',f'fixed_{index:02}_040','paper_eval.py',['--checkpoint',str(out/'epoch_040.pth'),'--force-plan',str(index)],[str(out/'epoch_040.pth')],rank=6)
    return dict(recipe=GRAPH_RECIPE,configs=rows,stages=stages,main_courses=4,mechanism_courses=3,seed=42,
                epochs_main=80,epochs_mechanism=40,performance_gates=False,existing_configurations_preserved=79,
                total_configurations=86,total_training_courses=84,total_stages=374+len(stages))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime-root',default='/data/run01/sczc063/yuzibo/h65_clean_adatad_20260910/graph_tad_20260914');a=p.parse_args()
    plan=build(a.runtime_root)
    for cfg in plan['configs']:json_write(ROOT/'configs/graph'/f'{cfg["id"]}.json',cfg)
    json_write(ROOT/'research/paper/graph/plan.json',plan)
    print(json.dumps(dict(configurations=len(plan['configs']),stages=len(plan['stages']),registered_with_owner=False)))
