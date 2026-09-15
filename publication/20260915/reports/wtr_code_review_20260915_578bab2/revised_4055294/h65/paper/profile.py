"""Actual matrix/conv accounting and GPU-resident model timings."""
import copy
import time
import numpy as np
import torch
from h65.frame.measure import matrix_counter
from .interventions import evaluation_state
from .geometry import candidate_mask


def linear_coefficient(module):
    return sum(m.in_features*m.out_features for m in module.modules() if isinstance(m,torch.nn.Linear))


def encoder_macs(model,trace):
    vit=model.encoder.vit;c=vit.embed_dims;shape=trace['patch_input_shape']
    p=(shape[-2]//vit.patch_size)*(shape[-1]//vit.patch_size)
    rows=shape[0]*8*p
    patch=vit.patch_embed.projection
    total=rows*patch.out_channels*int(np.prod(patch.weight.shape[1:]))
    for i,block in enumerate(vit.blocks):
        total+=(2*trace['q'][i]+2*trace['kv'][i])*c*c+trace['qk_av_macs'][i]+trace['score_qk'][i]
        total+=trace.get('graph_router_macs',[0]*len(vit.blocks))[i]
        total+=trace.get('operator_value_macs',[0]*len(vit.blocks))[i]
        total+=trace['heavy_mlp'][i]*linear_coefficient(block.mlp)
        total+=trace['light'][i]*linear_coefficient(model.encoder.engine.light[i])
        if hasattr(model.encoder.engine,'depth_attention'):
            total+=trace.get('depth_attention_light',[0]*len(vit.blocks))[i]*linear_coefficient(model.encoder.engine.depth_attention[i])
            total+=trace.get('depth_ffn_light',[0]*len(vit.blocks))[i]*linear_coefficient(model.encoder.engine.depth_ffn[i])
        if block.use_adapter:
            coefficient=linear_coefficient(block.adapter)
            coefficient+=sum(m.out_channels*(m.in_channels//m.groups)*int(np.prod(m.kernel_size))
                             for m in block.adapter.modules() if isinstance(m,torch.nn.Conv1d))
            total+=trace['tia'][i]*coefficient
    return int(total)


def execution_flops(model,detail):
    index=detail['plan_index']
    if index is None:raise ValueError('Dataset compute ledger requires a registered plan')
    constant=float(model.budget_router.fixed_nonencoder_macs[index])
    if not np.isfinite(constant):raise RuntimeError('Missing validated execution-cost ledger')
    pair_cost=detail['routing'].get('pair_count',0)*linear_coefficient(model.frame_router.network)
    return 2*(constant+encoder_macs(model,detail['trace'])+pair_cost+detail['trace'].get('graph_recovery_macs',0))


def _hooks(model,counter):
    handles=[]
    def add(module,name):
        def enter(m,a):counter.phase.append(name)
        def leave(m,a,o):counter.phase.pop()
        handles.extend((module.register_forward_pre_hook(enter),module.register_forward_hook(leave)))
    add(model.encoder.scout,'scout');add(model.encoder.engine,'backbone')
    add(model.decoder,'decoder');add(model.budget_router.network,'budget_router');add(model.frame_router.network,'frame_router')
    if hasattr(model,'graph_recovery'):add(model.graph_recovery,'graph_recovery')
    if hasattr(model.encoder.engine,'graph_attention'):
        for module in model.encoder.engine.graph_attention.values():
            add(module,'graph_attention');add(module.router,'graph_router')
    for block in model.encoder.vit.blocks:
        add(block.mlp,'heavy_ffn')
        if block.use_adapter:add(block.adapter,'tia')
    for light in model.encoder.engine.light:add(light,'light_ffn')
    if hasattr(model.encoder.engine,'depth_attention'):
        for light in model.encoder.engine.depth_attention:add(light,'depth_light_attention')
        for light in model.encoder.engine.depth_ffn:add(light,'depth_light_ffn')
    detector=model.readout.detector
    for name in ('projection','neck'):
        if hasattr(detector,name):add(getattr(detector,name),'student_head')
    if hasattr(detector,'rpn_head'):
        for name in ('cls_head','reg_head','cls_convs','reg_convs'):
            if hasattr(detector.rpn_head,name):
                for module in getattr(detector.rpn_head,name).modules():
                    if isinstance(module,(torch.nn.Linear,torch.nn.Conv1d,torch.nn.MultiheadAttention)):add(module,'student_head')
    else:
        for module in detector.transformer.modules():
            if isinstance(module,(torch.nn.Linear,torch.nn.Conv1d,torch.nn.MultiheadAttention)):add(module,'student_head')
    return handles


@torch.no_grad()
def measure(model,data,force_plan=None,repeats=20):
    torch.cuda.synchronize();baseline_bytes=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
    with evaluation_state(model),torch.autocast('cuda',dtype=torch.bfloat16):
        counter=matrix_counter();hooks=_hooks(model,counter)
        try:
            with counter:prediction,detail=model.predictions(data,force_plan)
        finally:
            for hook in hooks:hook.remove()
        missing=counter.unresolved_matrix_ops()
        if missing:raise RuntimeError('Uncounted matrix operations: '+str(missing))
        latencies=[]
        if repeats:
            for _ in range(3):model.predictions(data,force_plan)
            torch.cuda.synchronize()
            for _ in range(repeats):
                start=time.perf_counter();model.predictions(data,force_plan);torch.cuda.synchronize()
                latencies.append((time.perf_counter()-start)*1000)
        macs=dict(counter.macs);route_qk=sum(detail['trace']['routing_qk'])
        if route_qk:
            macs['backbone']=macs.get('backbone',0)-route_qk;macs['attention_routing_qk']=route_qk
        mask=candidate_mask(data)
        return dict(matrix_conv_flops=2*sum(counter.macs.values()),macs_by_component=macs,
                    operations=dict(counter.operations),fused_attention=counter.fused_attention,
                    valid_candidates=int(mask.sum()),physical_candidates=detail['plan']['frames'],
                    raw_model_latency_ms=latencies,latency_mean_ms=float(np.mean(latencies)) if latencies else None,
                    latency_median_ms=float(np.median(latencies)) if latencies else None,
                    latency_p95_ms=float(np.percentile(latencies,95)) if latencies else None,
                    peak_gib=torch.cuda.max_memory_allocated()/2**30,
                    incremental_peak_gib=(torch.cuda.max_memory_allocated()-baseline_bytes)/2**30,
                    resident_gib=baseline_bytes/2**30,external_teacher_resident=model.teacher is not None,
                    support_reference_resident=getattr(model,'support_reference',None) is not None,
                    memory_scope='peak allocated in this process; incremental peak subtracts resident inputs, weights and any training states',
                    scope='resident RGB; scout/router/encoder/TIA/light/decoder/student head; excludes decoding/NMS',
                    matrix_scope='2 MAC; convolution, linear, matrix products and QK/AV; non-matrix arithmetic excluded',
                    plan=detail['trace']['plan'],encoder_macs_from_trace=encoder_macs(model,detail['trace']),
                    frame_router_pair_count=detail['routing'].get('pair_count',0),
                    graph_recovery_macs_from_trace=detail['trace'].get('graph_recovery_macs',0),
                    graph_execution={key:detail['trace'].get(key) for key in ('graph_router_macs','graph_candidate_slots','graph_referral_paths','graph_retained_edges')},
                    execution_by_layer={k:detail['trace'][k] for k in ('q','kv','heavy_mlp','light','depth_attention_light','depth_ffn_light','tia','score_qk')})


@torch.no_grad()
def calibrate_costs(model,data):
    with evaluation_state(model):return _calibrate_costs(model,data)


def _calibrate_costs(model,data):
    if int(candidate_mask(data).sum())!=768:raise ValueError('Primary cost table requires a full 768-candidate window')
    costs=[];records=[];cache={};constants=[]
    for index in range(len(model.menu)):
        plan=model.plan(0 if model.config.get('dense_baseline',False) else index)
        key=tuple(sorted((k,str(v)) for k,v in plan.items() if k not in ('id','nominal_id')))
        if key not in cache:cache[key]=measure(model,data,index,repeats=0)
        record=copy.deepcopy(cache[key]);records.append(record)
        measured_encoder=sum(record['macs_by_component'].get(k,0) for k in ('backbone','heavy_ffn','light_ffn','depth_light_attention','depth_light_ffn','tia','attention_routing_qk','graph_attention','graph_router'))
        if abs(measured_encoder-record['encoder_macs_from_trace'])>1:
            raise RuntimeError(f'Execution ledger disagrees with operator count: {measured_encoder} vs {record["encoder_macs_from_trace"]}')
        pairs=record['frame_router_pair_count']*linear_coefficient(model.frame_router.network)
        graph=record['graph_recovery_macs_from_trace']
        if abs(record['macs_by_component'].get('graph_recovery',0)-graph)>1:
            raise RuntimeError('Temporal graph ledger disagrees with actual operators')
        constants.append(record['matrix_conv_flops']/2-record['encoder_macs_from_trace']-pairs-graph)
        # Conservative allowance for content-dependent local frame-pair counts.
        allowance=(2*linear_coefficient(model.frame_router.network)*(768-plan['frames'])/1e9 if model.config.get('frame_utility',True) else 0.)
        costs.append(record['matrix_conv_flops']/1e9+allowance)
    saved=dict(model.config);resolution=model.encoder.resolution
    model.config['dense_baseline']=True;model.config['static_depth']=None;model.encoder.resolution=160
    try:reference=measure(model,data,0,repeats=0)
    finally:model.config=saved;model.encoder.resolution=resolution
    model.budget_router.cost_gflops.copy_(torch.tensor(costs,device=model.budget_router.cost_gflops.device))
    model.budget_router.reference_gflops.fill_(reference['matrix_conv_flops']/1e9)
    model.budget_router.fixed_nonencoder_macs.copy_(torch.tensor(constants,dtype=torch.float64,device=model.budget_router.fixed_nonencoder_macs.device))
    return dict(plans=records,conservative_nominal_gflops=costs,reference=reference,
                budget_scope='full-window same-backbone/same-head dense reference',
                allowance='at most T-K frame-pair MLP rows, added conservatively to measured cost')
