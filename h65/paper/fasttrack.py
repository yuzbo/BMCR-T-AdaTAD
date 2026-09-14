"""Fast-Track initialization and evidence-based course admission."""
from pathlib import Path
import json
import torch


def initialize(model,resources):
    path=resources['wtr_initialization']
    payload=torch.load(path,map_location='cpu')
    current=model.state_dict();restored=[]
    for key,value in payload['ema'].items():
        if key in current:
            if current[key].shape!=value.shape:raise ValueError('V2 initialization shape differs: '+key)
            current[key].copy_(value.to(current[key]));restored.append(key)
    if not any(k.startswith('decoder.') for k in restored) or not any(k.startswith('readout.') for k in restored):
        raise ValueError('Fast-Track needs the complete frozen V2 source, not a light-only subset')
    return dict(asset=path,source_checkpoint=payload['original_checkpoint'],source_revision=payload['metadata']['source_revision'],
        restored_tensors=len(restored),new_operator_value_heads=True,optimizer_resumed=False,
        course='80 new adaptation epochs from shared V2-S epoch40 EMA initialization')


def admission(cfg,resources):
    review_path=Path(resources['wtr_review_directory'])/(cfg['id']+'.json')
    if not review_path.exists():raise RuntimeError(f'Cross-review is pending: {review_path}')
    review=json.loads(review_path.read_text())
    sha=(Path(__file__).resolve().parents[2]/'WTR_FASTTRACK_SCIENCE_SHA').read_text().strip()
    if not (review.get('passed') and review.get('scope')=='code_and_protocol_review'
            and review.get('config_id')==cfg['id'] and review.get('science_sha')==sha
            and review.get('reviewer_thread_id') and review.get('evidence')):
        raise RuntimeError('Cross-review does not cover this experiment and scientific revision')
    requirement=cfg.get('requires_evidence')
    if not requirement:return
    path=Path(resources['wtr_gate_directory'])/(requirement+'.json')
    if not path.exists():raise RuntimeError(f'Course remains WAITING for development evidence: {path}')
    receipt=json.loads(path.read_text())
    if not receipt.get('passed') or receipt.get('split')!='development' or not receipt.get('evidence'):
        raise RuntimeError('Course admission needs a passed development gate with actual evidence')


def binding(model,metadata,updates):
    return dict(course_id=model.config['id'],science_sha=metadata['wtr_fasttrack_science_sha'],
        source_revision=metadata['source_revision'],parameter_state=f"{model.config['id']}:pre_update:{updates}",
        initialization=metadata['wtr_initialization'],successful_updates=updates,
        ds_policy_version=model.config['operator_policy_version'],ds_policy=model.config['operator_policy'],
        recovery_version='V2-S Cross multidepth; current course parameters',
        light_version='V2-S initialized light attention/FFN; current course parameters',graph_context_version='disabled',
        use='online immediate supervision; persisted checkpoint action banks are re-queried separately')


def router_splits(resources):
    splits=resources['wtr_router_splits']
    groups=[set(splits[k]) for k in ('fit','calibration','holdout')]
    if [len(g) for g in groups]!=[160,20,20] or any(groups[i]&groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Invalid router-label development partition')
    if set.union(*groups)!=set(resources['datasets']['thumos']['train_ids']):
        raise ValueError('Router-label partition differs from the training population')
    return splits
