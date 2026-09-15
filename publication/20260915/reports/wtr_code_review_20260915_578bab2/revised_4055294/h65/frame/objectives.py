"""Actual original-grid task gradients and separately controlled latent/KD losses."""
import torch
import torch.nn.functional as F

def native_weights(data):
    masks=data['masks'];valid=masks.reshape(len(masks),-1,2).any(-1);weight=valid.float()
    if 'gt_boundary_validity' in data:
        centers=torch.arange(valid.shape[1],device=masks.device).float()*2+.5
        for row,(boxes,flags) in enumerate(zip(data['gt_segments'],data['gt_boundary_validity'])):
            endpoints=boxes.flatten()[flags.flatten()]
            if len(endpoints):weight[row]*=1+2*torch.exp(-.5*((centers[:,None]-endpoints[None])/2).square()).amax(1)
    return weight,valid

def feature_distance(student,target,weight):
    student,target=student.float(),target.detach().float()
    error=F.smooth_l1_loss(student,target,reduction='none').mean(1)
    cosine=1-F.cosine_similarity(student,target,dim=1,eps=1e-6)
    return ((error+.05*cosine)*weight).sum()/weight.sum().clamp_min(1)

def bernoulli_kl(student_logits,teacher_logits,valid,temperature=2.):
    logits=student_logits.float()/temperature;target=(teacher_logits.detach().float()/temperature).sigmoid().clamp(1e-6,1-1e-6)
    cross=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
    entropy=-(target*target.log()+(1-target)*(1-target).log())
    return (((cross-entropy).mean(-1))*valid).sum()/valid.sum().clamp_min(1)*temperature**2

def training_objectives(student_native,teacher_native,data,teacher,feature_weight=1.,gt_weight=1.,difference_weight=0.,output_kd_weight=0.):
    weight,valid=native_weights(data);zero=student_native.float().sum()*0
    feature=feature_distance(student_native,teacher_native,weight)
    cls=reg=gt=zero
    if gt_weight:
        raw=teacher.loss(student_native,data)
        cls,reg=raw['cls_loss'],raw['reg_loss'];gt=raw['cost']
    difference=zero
    if difference_weight:
        pair=valid[:,1:]&valid[:,:-1]
        error=F.smooth_l1_loss(student_native.float().diff(dim=-1),teacher_native.detach().float().diff(dim=-1),reduction='none').mean(1)
        difference=(error*pair).sum()/pair.sum().clamp_min(1)
    kd=zero
    if output_kd_weight:
        student=teacher.raw_head(student_native,data['masks'])
        with torch.no_grad():target=teacher.raw_head(teacher_native,data['masks'])
        kd=bernoulli_kl(student['logits'],target['logits'],student['valid'])
        confidence=target['logits'].sigmoid().amax(-1)*student['valid']
        error=F.smooth_l1_loss(student['offsets'],target['offsets'].detach(),reduction='none').mean(-1)
        kd=kd+.1*(error*confidence).sum()/confidence.sum().clamp_min(1)
    cost=gt_weight*gt+feature_weight*feature+difference_weight*difference+output_kd_weight*kd
    return dict(cost=cost,cls_loss=cls,reg_loss=reg,gt_loss=gt,feature_loss=feature,difference_loss=difference,output_kd_loss=kd)

def shared_full_student_stopgrad(student_native,teacher_native,data,*,label_source='self',feature_weight=1.):
    if label_source not in ('external','frozen_anchor','self'):raise ValueError(label_source)
    weight,_=native_weights(data)
    return feature_weight*feature_distance(student_native,teacher_native,weight)
