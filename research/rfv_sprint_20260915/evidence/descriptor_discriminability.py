"""One fixed observational descriptor diagnostic; no learner, router, or CF queries."""
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path('/root/autodl-tmp/rfv_20260915')
CONSUMER = 'b679ae0ec46510bd85dc303dfacd4b6a6fd5288d'
sys.path[:0] = [str(ROOT/'versions'/CONSUMER), str(ROOT/'versions'/CONSUMER/'upstream')]
from h65.rfv.dataset import load_bank

torch.set_num_threads(2)
rows, binding = load_bank([ROOT/'results/2ca4d4b'/f'mini_c40_shard{i}' for i in (0,1)])
diagnostic_dir = ROOT/'results/d62ea55/ACTION_HOLDOUT'
prior = json.loads((diagnostic_dir/'ACTION_HOLDOUT_DIAGNOSTIC.json').read_text())
checkpoint_path = diagnostic_dir/'plain_m_action_holdout_s42.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
scaler = checkpoint['snapshot']['state']
input_mean = scaler['input_mean'].float()
input_scale = scaler['input_scale'].float()
assert len(input_mean)==407 and bool((input_scale>0).all())

N = 10000
rng = np.random.default_rng(42)
records=[]
per_video={}
exact_collisions=[]

for row in rows:
    if row['partition']!='fit' or not row['action_pairs']:
        continue
    assert len(row['action_pairs'])==16
    order=sorted(range(16),key=lambda i:(row['action_pairs'][i][1],row['action_pairs'][i][0]))
    fit=np.asarray(order[::2]);held=np.asarray(order[1::2])
    ids=[action['id'] for action in row['actions']]
    original=prior['action_ids'][row['state_key']]
    assert [ids[i] for i in fit]==original['fit_action_ids']
    assert [ids[i] for i in held]==original['held_action_ids']
    # Exactly the saved model's float32 input normalization; no held-data refit.
    x=torch.from_numpy(row['arrays']['descriptor']).float()
    z=((x-input_mean)/input_scale).numpy().astype(np.float64)
    y=row['arrays']['target'].sum(-1).astype(np.float64)
    noise=max(1e-8,3*max(row['no_op_error'],row['replay_max_error']))
    spread=float(np.ptp(y))
    assert spread>noise
    clear=np.abs(y)>noise
    # Previously observed fit data have no ambiguous labels. Do not invent a
    # sign convention if a different bank is supplied.
    assert bool(clear.all()), 'The fixed diagnostic requires its declared clear-sign fit bank'
    constant=np.ptp(z,axis=0)==0
    equal=(z[:,None,:]==z[None,:,:]).all(-1)
    for i,j in zip(*np.where(np.triu(equal,1))):
        exact_collisions.append(dict(video_id=row['video_id'],state_key=row['state_key'],
            action_ids=[ids[i],ids[j]],gain=[float(y[i]),float(y[j])],
            noise_epsilon=noise,material_conflict=bool(abs(y[i]-y[j])>2*noise)))

    distance2=((z[held,None,:]-z[None,fit,:])**2).mean(-1)
    nearest=distance2==distance2.min(-1,keepdims=True)
    weight=nearest/nearest.sum(-1,keepdims=True)
    yh=y[held];yf=y[fit]
    discrepancy=np.abs(yh[:,None]-yf[None,:])/spread
    disagreement=np.sign(yh[:,None])!=np.sign(yf[None,:])
    near_magnitude=float((discrepancy*weight).sum(-1).mean())
    random_magnitude=float(discrepancy.mean())
    near_sign=float((disagreement*weight).sum(-1).mean())
    random_sign=float(disagreement.mean())
    observed_mag=near_magnitude-random_magnitude
    observed_sign=near_sign-random_sign

    # Shuffle only the association of the eight fit gains to the fixed inputs.
    # Neighbors, held gains, supports, scalers and all model parameters stay fixed.
    permutations=np.argsort(rng.random((N,8)),axis=1)
    shuffled=yf[permutations]
    shuffled_discrepancy=np.abs(yh[None,:,None]-shuffled[:,None,:])/spread
    null_mag=(shuffled_discrepancy*weight[None]).sum(-1).mean(-1)-random_magnitude
    shuffled_disagreement=np.sign(yh[None,:,None])!=np.sign(shuffled[:,None,:])
    null_sign=(shuffled_disagreement*weight[None]).sum(-1).mean(-1)-random_sign
    per_video.setdefault(row['video_id'],[]).append((observed_mag,observed_sign,null_mag,null_sign))

    pairs=[]
    for h_index,action in enumerate(held):
        candidates=np.flatnonzero(nearest[h_index])
        pairs.append(dict(held_action=ids[action],held_gain=float(y[action]),
            nearest_fit_actions=[ids[fit[j]] for j in candidates],
            nearest_fit_gains=[float(y[fit[j]]) for j in candidates],
            standardized_rms_distance=float(np.sqrt(distance2[h_index].min())),
            uniform_mean_distance=float(np.sqrt(distance2[h_index]).mean()),
            sign_disagreement=float((disagreement[h_index]*weight[h_index]).sum()),
            normalized_gain_difference=float((discrepancy[h_index]*weight[h_index]).sum())))
    records.append(dict(video_id=row['video_id'],state_key=row['state_key'],actions=16,
        gain_range=spread,noise_epsilon=noise,constant_dimensions=int(constant.sum()),
        support_global_blocks_constant=bool(constant[192:384].all()),
        near_normalized_gain_difference=near_magnitude,random_normalized_gain_difference=random_magnitude,
        difference_vs_random=observed_mag,near_sign_disagreement=near_sign,
        random_sign_disagreement=random_sign,sign_difference_vs_random=observed_sign,
        exact_tie_queries=int((nearest.sum(-1)>1).sum()),pairs=pairs))

observed=[];nulls=[]
for video in sorted(per_video):
    values=per_video[video]
    observed.append(np.mean([[v[0],v[1]] for v in values],axis=0))
    nulls.append(np.mean([np.stack((v[2],v[3]),-1) for v in values],axis=0))
observed=np.asarray(observed);null=np.mean(nulls,axis=0)
bootstrap_rng=np.random.default_rng(43)
draws=observed[bootstrap_rng.integers(0,len(observed),(N,len(observed)))].mean(1)
statistics={}
for i,name in enumerate(('normalized_gain_difference_vs_random','sign_disagreement_vs_random')):
    value=float(observed[:,i].mean())
    statistics[name]=dict(mean=value,video_bootstrap_ci95=np.quantile(draws[:,i],[.025,.975]).tolist(),
        permutation_null_ci95=np.quantile(null[:,i],[.025,.975]).tolist(),
        permutation_null_mean=float(null[:,i].mean()),
        one_sided_p_for_more_local_consistency=float((1+np.count_nonzero(null[:,i]<=value))/(N+1)),
        favorable_videos=int((observed[:,i]<0).sum()),videos=len(observed))

report=dict(scope='observational within-state descriptor association; not a router or a scientific gate',
    source_bank=binding,scaler_checkpoint=str(checkpoint_path),scaler_scope='saved seed42 fit8 input_mean/input_scale',
    split='exact original physical-order 8 fit / 8 held actions in existing mini fit videos',
    metric='RMS Euclidean distance in actual float32-normalized full407; exact nearest ties averaged equally',
    magnitude_scale='same-state actual cls+loc max-min; diagnostic only, not fitting or inference',
    null='10000 permutations of eight fit-label assignments independently within each state; held labels fixed',
    bootstrap='10000 paired video resamples; fixed 25-video diagnostic cohort, not a training-seed interval',
    permutation_seed=42,bootstrap_seed=43,primary_statistic='normalized_gain_difference_vs_random',
    secondary_statistic='sign_disagreement_vs_random',videos=len(per_video),states=len(records),
    held_queries=sum(len(r['pairs']) for r in records),exact_equal_input_pairs=len(exact_collisions),
    exact_material_conflicts=sum(r['material_conflict'] for r in exact_collisions),
    exact_collisions=exact_collisions,statistics=statistics,
    descriptive=dict(near_normalized_gain_difference=float(np.mean([r['near_normalized_gain_difference'] for r in records])),
        random_normalized_gain_difference=float(np.mean([r['random_normalized_gain_difference'] for r in records])),
        near_sign_disagreement=float(np.mean([r['near_sign_disagreement'] for r in records])),
        random_sign_disagreement=float(np.mean([r['random_sign_disagreement'] for r in records])),
        all_support_global_blocks_constant=all(r['support_global_blocks_constant'] for r in records),
        exact_tie_queries=sum(r['exact_tie_queries'] for r in records)),
    records=records,new_cf_queries=0,optimizer_updates=0,model_forward_evaluations=0,
    outer_holdout_used=False,calibration_used=False,scientific_gate=False,
    interpretation_limits=['Full407 includes action coordinates: no exact duplicates does not establish sufficiency.',
        'Near-input disagreement tests local consistency only under this fixed metric, not all possible functions.',
        'No nearest-neighbor predictor, routing policy, threshold search, or regret result is constructed.'])
pairs=[pair for row in records for pair in row['pairs']]
near_distances=np.array([p['standardized_rms_distance'] for p in pairs])
random_distances=np.array([p['uniform_mean_distance'] for p in pairs])
quantiles=[0,.25,.5,.75,1]
report['descriptive']['distance_summary']=dict(quantiles=quantiles,
    nearest_standardized_rms_distance=np.quantile(near_distances,quantiles).tolist(),
    random_mean_standardized_rms_distance=np.quantile(random_distances,quantiles).tolist(),
    nearest_over_random_mean_distance=np.quantile(near_distances/random_distances,quantiles).tolist(),
    meaning='Nearest among the eight fixed fit actions; no absolute nearness threshold is claimed')
print(json.dumps(report,allow_nan=False))
