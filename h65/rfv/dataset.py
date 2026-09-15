"""Read complete shared banks and expose only declared predictor inputs."""
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from h65.raw.value import FEATURE_DIM

INPUTS=('descriptor','cheap','times','node_valid','actionness','transition','support_distance',
        'support_occupancy','remove_times','insert_times','support_times','support_valid')


def require_equivalent_captures(bindings,receipt_path=None):
    revisions={revision for binding in bindings for revision in binding['capture_revisions']}
    if len(revisions)==1:return dict(capture_revisions=sorted(revisions),same_revision=True)
    if receipt_path is None:raise ValueError('Different capture revisions require a reviewed label-forward equivalence receipt')
    receipt=json.loads(Path(receipt_path).read_text())
    if not (receipt.get('passed') and receipt.get('scope')=='rfv_label_forward_equivalence'
            and set(receipt.get('capture_revisions',[]))==revisions and receipt.get('evidence')):
        raise ValueError('Equivalence receipt does not cover these exact capture revisions')
    return receipt


def parameter_key(identity):
    return {key:identity[key] for key in ('original_checkpoint','parameter_state','epoch','updates','source_revision','replay_policy')}


def load_bank(paths):
    manifests=[];rows=[];seen=set();shards=set();count=None;protocol=None;parameter=None
    for folder in map(Path,paths):
        local_manifests=sorted(folder.glob('manifest_*.json'))
        if not local_manifests:raise ValueError('Missing bank manifest')
        declared=set()
        for file in local_manifests:
            meta=json.loads(file.read_text());shard=meta['shard']
            if meta['mode'] not in ('bank','replay'):raise ValueError('Use the registered common fixed-action bank')
            if shard in shards:raise ValueError('Duplicate bank shard')
            shards.add(shard)
            if count is None:count=meta['shards'];protocol=meta['protocol'];parameter=parameter_key(meta['checkpoint']);cohort=meta['cohort']
            if meta['shards']!=count or meta['protocol']!=protocol or parameter_key(meta['checkpoint'])!=parameter or meta['cohort']!=cohort:
                raise ValueError('Bank shards differ in protocol/cohort/parameter identity')
            done=folder/f'completed_{shard}.json'
            if not done.exists():raise ValueError('Bank shard is incomplete')
            receipt=json.loads(done.read_text())
            if receipt['source_revision']!=meta['source_revision'] or receipt['checkpoint']!=meta['checkpoint']:
                raise ValueError('Completion does not identify this capture')
            declared.update(receipt['group_files']);manifests.append(meta)
        if {x.name for x in (folder/'groups').glob('*.json')}!=declared:
            raise ValueError('Bank group files differ from the completed inventory')
        for name in sorted(declared):
            path=folder/'groups'/name;row=json.loads(path.read_text())
            if row['state_key'] in seen:raise ValueError('Duplicate state across shards')
            seen.add(row['state_key'])
            with np.load(path.parent/row['arrays_file'],allow_pickle=False) as stored:
                arrays={key:stored[key].copy() for key in (*INPUTS,'target')}
            n=len(row['action_pairs'])
            if arrays['descriptor'].shape!=(n,FEATURE_DIM) or arrays['target'].shape!=(n,2):
                raise ValueError('Bank descriptor/target shape differs')
            target=np.asarray([x['gain_cls_loc'] for x in row['actions']],dtype=np.float32).reshape(-1,2)
            if not np.array_equal(arrays['target'],target):raise ValueError('JSON and array actual gains differ')
            if any(not np.isfinite(value).all() for value in arrays.values()):raise ValueError('Nonfinite bank array')
            row['arrays']=arrays;rows.append(row)
    if shards!=set(range(count)):raise ValueError('All predeclared shards must complete')
    groups=protocol['mini' if cohort=='mini' else 'splits']
    expected={video:partition for partition,ids in groups.items() for video in ids}
    if {row['video_id'] for row in rows}!=set(expected):raise ValueError('Incomplete registered bank video cohort')
    if any(row['partition']!=expected[row['video_id']] for row in rows):raise ValueError('Bank partition role differs')
    if cohort=='mini' and set(expected)&set(protocol['splits']['holdout']):raise ValueError('Mini consumed sealed outer holdout')
    return rows,dict(protocol=protocol,cohort=cohort,parameter=parameter,
        capture_revisions=sorted({x['source_revision'] for x in manifests}),
        note='capture revisions retained; consumers must review forward equivalence before mixing revisions')


def collate_states(rows,device):
    maximum=max(len(row['action_pairs']) for row in rows)
    batch={};action_inputs={'descriptor','remove_times','insert_times','target'}
    for key in (*INPUTS,'target'):
        tensors=[]
        for row in rows:
            value=torch.from_numpy(row['arrays'][key])
            if key in action_inputs and len(value)<maximum:
                value=torch.cat((value,torch.zeros((maximum-len(value),*value.shape[1:]),dtype=value.dtype)))
            tensors.append(value)
        batch[key]=torch.stack(tensors).to(device)
    batch['action_valid']=torch.stack([torch.arange(maximum)<len(row['action_pairs']) for row in rows]).to(device)
    return batch


def normalization(rows):
    fit=[row for row in rows if row['partition']=='fit' and len(row['action_pairs'])]
    if not fit:raise ValueError('No fit actions')
    descriptor=torch.cat([torch.from_numpy(x['arrays']['descriptor']) for x in fit])
    target=torch.cat([torch.from_numpy(x['arrays']['target']) for x in fit])
    cheap=torch.cat([torch.from_numpy(x['arrays']['cheap'][x['arrays']['node_valid']]) for x in fit])
    if not bool(target.abs().max()>1e-8):raise ValueError('No fit signal above numerical floor')
    return descriptor,target,cheap


def bank_identities(rows):
    """Requested v3 identities for the saved input view; no target enters either."""
    result=[]
    for row in rows:
        state={key:row[key] for key in ('video_id','state_key','episode','selected_frame_ids',
            'selected_valid','candidate_frame_ids','continuation')}
        digest=hashlib.sha256(json.dumps(state,sort_keys=True,separators=(',',':')).encode())
        for key in INPUTS:
            array=np.ascontiguousarray(row['arrays'][key])
            digest.update(json.dumps([key,str(array.dtype),list(array.shape)],separators=(',',':')).encode())
            digest.update(array.tobytes())
        candidates=dict(pairs=row['action_pairs'],ids=[item['id'] for item in row['actions']])
        candidate_hash=hashlib.sha256(json.dumps(candidates,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        result.append(dict(state_key=row['state_key'],video_id=row['video_id'],partition=row['partition'],
            state_hash=digest.hexdigest(),candidate_hash=candidate_hash))
    return dict(states=result,state_hash_scope='physical/support metadata and every declared predictor input; no labels',
        candidate_hash_scope='ordered physical pairs and their fixed IDs; no labels',
        checkpoint_scope='checkpoint recorded separately; learned cheap states may differ across checkpoints',
        use='R1 input-view identity and RISE same-state function binding; never overwrite the captured bank')
