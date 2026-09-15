#!/usr/bin/env python3
"""Small real S/S-prime return check and exact768-cheap reverse-view validation."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]
import numpy as np
import torch
from h65.paper.runtime import read_config,json_write
from h65.paper.geometry import candidate_mask
from h65.paper.temporal_value import public_state,to_standard
from h65.full.runtime import to_gpu
from h65.atlas.data import CharacterizationData
from h65.atlas.reference import deterministic_fp32
from h65.rfv.hardware import initialize
from h65.rfv.bank import FixedTModel
from h65.rfv.dataset import load_bank,parameter_key
from h65.rfv.reverse import reverse_descriptors
from h65.raw.contracts import RawSelection,swap
from h65.raw.value import descriptors


def normalizer(model):
    return float(model.readout.detector.rpn_head.loss_normalizer)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bank',action='append',required=True)
    parser.add_argument('--resources',required=True);parser.add_argument('--config',default=str(ROOT/'configs/rfv/T-U.json'))
    parser.add_argument('--normalization-checkpoint',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    revision=(ROOT/'source_revision.txt').read_text().strip()
    rows,binding=load_bank(args.bank)
    fit=sorted([r for r in rows if r['partition']=='fit' and r['action_pairs']],key=lambda r:r['state_key'])
    full=[r for r in fit if sum(r['episode']['official_valid'])==768]
    tail=[r for r in fit if sum(r['episode']['official_valid'])<768]
    selected=[full[0]]
    if tail:selected.append(tail[0])
    selected.extend(r for r in fit if r['state_key'] not in {v['state_key'] for v in selected})
    selected=selected[:3]
    config=dict(source_revision=revision,bank=binding,states=[r['state_key'] for r in selected],
        pairs='first and last registered candidate per selected state',descriptor_atol=1e-6,
        normalized_descriptor_atol=1e-3,normalization_checkpoint=args.normalization_checkpoint,
        scope='six existing fit pairs at most; four independent full forwards per pair; no training labels added')
    target=out/'REVERSE_CONTRACT.json'
    if target.exists():
        prior=json.loads(target.read_text())
        if prior['config']!=config:raise ValueError('Existing reverse contract has different source/input identity')
        print(json.dumps(dict(passed=prior['passed'],already_complete=True)));return
    json_write(out/'config.json',config)
    resources=json.loads(Path(args.resources).read_text());hardware=initialize(resources)
    runtime=FixedTModel(read_config(args.config),resources)
    if parameter_key(runtime.identity)!=binding['parameter']:raise ValueError('Detector parameters differ from cached forward bank')
    plan=runtime.model.plan(runtime.cfg['fixed_plan'])
    if (plan['frames'],plan['depth'],plan['space'])!=(384,1.,1.):raise ValueError('Reverse mini only covers the fixed T K384/D100/S100 function')
    reference_head=torch.load(args.normalization_checkpoint,map_location='cpu',weights_only=False)
    if (reference_head['source_revision']!='b644d870d1845abbc1e4fd5ab7780f29ff96a53a' or
        reference_head['experiment']!=dict(suite='within',arm='plain_r0',bank=binding) or reference_head['seed']!=42):
        raise ValueError('Contract normalization must be the original within-fit8 P0 seed42')
    scaler=reference_head['snapshot']['state']['input_scale'].numpy()
    source=CharacterizationData(runtime.model_cfg,resources,'development',[r['video_id'] for r in selected],one_window=True)
    reference={(r['video_id'],r['window_start_frame']):r for r in selected};records=[];seen=set()
    with deterministic_fp32():
        for index in range(len(source)):
            data=to_gpu(source[index]);meta=data['metas'][0]
            ref=reference[(meta['video_name'],int(meta['window_start_frame']))];seen.add(ref['state_key'])
            preview,preview_cost=runtime.preview(data);seed=runtime.seed(data,preview);masks=candidate_mask(data)
            selection=to_standard(RawSelection(tuple(ref['selected_frame_ids']),tuple(ref['selected_valid'])),seed,
                tuple(meta['frame_inds'].tolist()),masks)
            episode,timeline,public,proposal,current=public_state(preview,selection,masks,data['metas'])
            if list(proposal.frame_ids)!=ref['candidate_frame_ids']:raise ValueError('Candidate domain changed')
            original=descriptors(episode,timeline,public,proposal,current,ref['action_pairs'],plan).cpu().numpy()
            forward_error=float(np.abs(original-ref['arrays']['descriptor']).max())
            reverse,views=reverse_descriptors(ref,ref['arrays']['descriptor'])
            for pair_index in sorted({0,len(ref['action_pairs'])-1}):
                remove,insert=ref['action_pairs'][pair_index]
                changed=swap(current,remove,insert,proposal);back=swap(changed,insert,remove,proposal)
                mapped=to_standard(changed,selection,episode.official_frame_ids,masks)
                returned=to_standard(back,selection,episode.official_frame_ids,masks)
                direct=descriptors(episode,timeline,public,proposal,changed,[(insert,remove)],plan).cpu().numpy()[0]
                error=reverse[pair_index]-direct
                before_norm=normalizer(runtime.model)
                losses=[];cost=[]
                for support in (selection,mapped,mapped,returned):
                    _,_,loss,flops=runtime.execute(data,support,preview,measure=True)
                    losses.append(loss.cpu().numpy());cost.append(flops)
                    if normalizer(runtime.model)!=before_norm:raise RuntimeError('Loss normalizer changed during fixed-function replay')
                forward=losses[0]-losses[1];inverse=losses[2]-losses[3]
                replay=max(float(np.abs(losses[0]-losses[3]).max()),float(np.abs(losses[1]-losses[2]).max()))
                epsilon=max(1e-8,3*replay)
                record=dict(state_key=ref['state_key'],forward_id=ref['actions'][pair_index]['id'],
                    inverse_id=f'{insert}->{remove}',losses_cls_loc=[v.tolist() for v in losses],
                    forward_gain=forward.tolist(),reverse_gain=inverse.tolist(),
                    gain_sum_max=float(np.abs(forward+inverse).max()),replay_max_error=replay,noise_epsilon=epsilon,
                    cached_gain_error=float(np.abs(forward-ref['arrays']['target'][pair_index]).max()),
                    forward_descriptor_error=forward_error,reverse_descriptor_error=float(np.abs(error).max()),
                    normalized_reverse_error=float(np.abs(error/scaler).max()),support_roundtrip=back==current,
                    normalizer_unchanged=True,full_forward_gflops=cost,preview_gflops=preview_cost,
                    direct_cheap_valid_nodes=int(public['hidden'].shape[1]),descriptor_dimension=int(direct.shape[-1]),
                    matched_body_cost=max(cost)-min(cost)<=1e-6)
                record['passed']=bool(record['gain_sum_max']<=epsilon and record['cached_gain_error']<=epsilon and
                    record['forward_descriptor_error']<=config['descriptor_atol'] and
                    record['reverse_descriptor_error']<=config['descriptor_atol'] and
                    record['normalized_reverse_error']<=config['normalized_descriptor_atol'] and
                    record['support_roundtrip'] and record['matched_body_cost'])
                records.append(record)
                print(json.dumps(dict(state=ref['state_key'],pair=record['forward_id'],passed=record['passed'],
                    descriptor_error=record['reverse_descriptor_error'],gain_sum=record['gain_sum_max'])),flush=True)
    if seen!={r['state_key'] for r in selected}:raise RuntimeError('Contract omitted a selected state')
    report=dict(config=config,hardware=hardware,checkpoint=runtime.identity,records=records,
        passed=all(r['passed'] for r in records),real_forward_calls=4*len(records),verified_pairs=len(records),
        new_training_labels=0,status='TECH_PASS' if all(r['passed'] for r in records) else 'TECH_FAIL')
    json_write(target,report);print(json.dumps(dict(status=report['status'],pairs=len(records))))


if __name__=='__main__':main()
