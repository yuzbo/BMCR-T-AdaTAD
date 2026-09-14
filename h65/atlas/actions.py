"""Locked primitive samples and finite groups made only of deployable primitives."""
import numpy as np
import torch

LAYERS=(4,6,8,10)
UNIFORM_ORDER=(0,5,10,15,2,7,8,13,1,6,11,12,3,4,9,14)
GROUP_COUNTS=(4,6,8,10,12,16)


def seed_for(meta,offset=0):
    return 42+offset+int(meta['window_index'])*1009+int(meta['video_ordinal'])*100003


def positions(valid,count=16):
    return np.floor((np.arange(count)+.5)*valid/count).clip(0,valid-1).astype(int).tolist()


def primitive_samples(data,meta):
    valid=int(data['masks'].sum());native=(valid+1)//2
    out=[]
    # Orthogonal deterministic strata cover every late layer, time and patch area.
    for i in range(16):
        layer=LAYERS[i%4]
        t=min(native-1,int(((i//4)+.5)*native/4))
        y=((i*3+meta['window_index'])%10)
        x=((i*7+meta['video_ordinal'])%10)
        out.append(dict(id=f'{layer}:{t}:{y}:{x}',layer=layer,native_time=t,y=y,x=x,
                        candidate_center=min(valid-1,2*t+.5)))
    return out


def full_masks(data):
    return torch.ones((12,384,10,10),device=data['inputs'].device,dtype=torch.bool)


def remove_points(data,axis,points):
    am=full_masks(data);fm=am.clone()
    for point in points:
        loc=(point['layer'],point['native_time'],point['y'],point['x'])
        fm[loc]=False
        if axis=='D':am[loc]=False
    return am,fm


def grouped_masks(data,axis,chosen):
    am=full_masks(data);fm=am.clone()
    valid=data['masks'][0].reshape(-1,2).any(-1)
    for j,layer in enumerate(LAYERS):
        if axis=='D':
            # Four pack-aligned physical time regions at each routed layer.
            for region in range(4):
                times=slice(region*96,(region+1)*96)
                if j*4+region not in chosen:
                    am[layer,times]=~valid[times,None,None]
                    fm[layer,times]=~valid[times,None,None]
        else:
            for region in range(4):
                y=slice((region//2)*5,(region//2+1)*5)
                x=slice((region%2)*5,(region%2+1)*5)
                if j*4+region not in chosen:
                    fm[layer,:,y,x]=~valid[:,None,None]
    return am,fm


def group_attention(dense,data,axis):
    scores=[];valid=data['masks'][0].reshape(-1,2).any(-1)
    for layer in LAYERS:
        score=dense['token_attention'][layer-1]
        for region in range(4):
            if axis=='D':
                part=score[region*96:(region+1)*96]
                mask=valid[region*96:(region+1)*96]
                values=part[mask]
            else:
                values=score[valid,(region//2)*5:(region//2+1)*5,(region%2)*5:(region%2+1)*5]
            scores.append(float(values.mean()) if values.numel() else 0.)
    return scores


def legal_prefix(order,costs,budget,mandatory=()):
    selected=list(mandatory);spent=0.
    for index in order:
        if index in selected:continue
        value=max(0.,float(costs[index]))
        if spent+value<=budget+1e-8:
            selected.append(int(index));spent+=value
    return selected
