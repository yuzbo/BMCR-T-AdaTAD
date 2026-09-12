"""Separate representation repair from real frame-computation interventions."""
import torch
import torch.nn.functional as F
from .router import candidate_pairs,swap_selection


def components(model,native,data):
    loss=model.teacher.loss(native,data)
    return torch.stack((loss['cls_loss'],loss['reg_loss']))


def paired_interventions(model,data,selection,output,teacher_native,max_pairs=2,measure_cost=False):
    pairs=candidate_pairs(output,selection,data['masks'],model.config.get('partner_scope','local'))
    if not pairs:return [],[],[]
    order=torch.randperm(len(pairs))[:max_pairs].tolist();pairs=[pairs[i] for i in order]
    was_training=model.training;model.eval();records=[];targets=[]
    try:
        with torch.no_grad():
            if measure_cost:
                from .measure import counted_native_and_loss
                native,loss,base_flops=counted_native_and_loss(model,data,selection)
                baseline=torch.stack((loss['cls_loss'],loss['reg_loss']))
            else:
                native,_=model.forward_native(data,selection=selection,apply_router=False)
                baseline=components(model,native,data)
            for row,remove,insert in pairs:
                changed=swap_selection(selection,row,remove,insert)
                if measure_cost:
                    counter,loss,changed_flops=counted_native_and_loss(model,data,changed)
                    actual=baseline-torch.stack((loss['cls_loss'],loss['reg_loss']))
                else:
                    counter,detail=model.forward_native(data,selection=changed,apply_router=False)
                    actual=baseline-components(model,counter,data)
                repaired=native.clone();positions=sorted({remove//2,insert//2})
                repaired[row,:,positions]=teacher_native[row,:,positions]
                repair=baseline-components(model,repaired,data)
                targets.append(actual)
                records.append(dict(row=row,video=data['metas'][row].get('video_name'),action_type='same_K_frame_swap',
                    remove=remove,insert=insert,repair_query_positions=positions,repair_delta=repair.cpu().tolist(),
                    actual_delta=actual.cpu().tolist(),baseline_components=baseline.cpu().tolist(),
                    actual_reencoded=True,teacher_kind='external_official',
                    selection_before=selection.indices[row,selection.valid[row]].cpu().tolist(),
                    selection_after=changed.indices[row,changed.valid[row]].cpu().tolist(),
                    note='repair is a representation proxy; action reruns selected RGB and all affected global states'))
                if measure_cost:records[-1].update(base_flops=base_flops,action_flops=changed_flops,actual_delta_flops=changed_flops-base_flops,
                    cost_scope='scout + frozen selection override + selected RGB encoder + decoder + GT head; excludes cached initial frame decision and teacher acquisition')
    finally:model.train(was_training)
    return pairs,torch.stack(targets),records


def utility_training_loss(model,data,detail,teacher_native,max_pairs=2):
    pairs,targets,records=paired_interventions(model,data,detail['selection'],detail['scout_output'],teacher_native,max_pairs)
    if not pairs:return teacher_native.sum()*0,records
    model.router.update_scales(targets)
    predicted=model.router.predict(detail['scout_output'],detail['selection'],data['masks'],pairs)
    for rec,value in zip(records,predicted.detach().cpu().tolist()):rec['predicted_normalized_delta']=value
    return .1*F.smooth_l1_loss(predicted.float(),targets.detach()/model.router.scales),records
