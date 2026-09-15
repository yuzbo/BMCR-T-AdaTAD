"""D1: complete teacher execution; no mixed sparse student graph in training."""
import torch
import torch.nn.functional as F
from .auxiliary import preview_images
from .routes import native_geometry


def boundary_weights(masks, segments, validity):
    valid=masks.reshape(len(masks),-1,2).any(-1)
    centers=torch.arange(valid.shape[-1],device=masks.device).float()*2+.5
    weights=valid.float().clone()
    for row,(boxes,flags) in enumerate(zip(segments,validity)):
        endpoints=boxes.flatten()[flags.flatten()]
        if endpoints.numel():
            distance=(centers[:,None]-endpoints[None]).abs()
            weights[row]*=1+2*torch.exp(-.5*(distance/2).square()).amax(1)
    return weights,valid


def feature_loss(prediction, target, weight, valid, spacing):
    prediction,target=prediction.float(),target.float()
    error=F.smooth_l1_loss(prediction,target,reduction='none').mean(1)
    cosine=1-F.cosine_similarity(prediction,target,dim=1,eps=1e-6)
    fit=((error+.1*cosine)*weight).sum()/weight.sum().clamp_min(1)
    pair=valid[:,1:]&valid[:,:-1]
    delta_prediction=(prediction[:,:,1:]-prediction[:,:,:-1])/spacing[:,None]
    delta_target=(target[:,:,1:]-target[:,:,:-1])/spacing[:,None]
    derivative=F.smooth_l1_loss(delta_prediction,delta_target,reduction='none').mean(1)
    return fit+.1*(derivative*pair).sum()/pair.sum().clamp_min(1)


def proxy_loss(logits, target, valid):
    target=target.detach().float()*valid
    mass=target.sum(-1,keepdim=True)
    probability=target/mass.clamp_min(1e-12)
    loss=-(probability*logits.float().masked_fill(~valid,-1e4).log_softmax(-1)).sum(-1)
    active=mass.squeeze(-1)>1e-12
    return loss[active].mean() if bool(active.any()) else logits.sum()*0


def dense_auxiliary_loss(model, data):
    masks=data['masks'];batch=len(masks)
    target,capture=model.teacher.dense_native(data['inputs'],capture=True)
    prediction0,utility_logits=model.aux.preview(preview_images(data['inputs']))
    prediction8=model.teacher.grid(model.aux.exit8(capture['exit8']))
    weights,valid=boundary_weights(masks,data['gt_segments'],data['gt_boundary_validity'])
    _,_,centers=native_geometry(masks,data['metas'])
    nominal=centers.new_tensor([2*meta.get('snippet_stride',1) for meta in data['metas']])[:,None]
    spacing=((centers[:,1:]-centers[:,:-1])/nominal).clamp_min(.25)
    losses=dict(feature0=feature_loss(prediction0,target,weights,valid,spacing),
                feature8=feature_loss(prediction8,target,weights,valid,spacing))

    # Only teacher final features require gradients here, not its parameters.
    # Detector stays eval/frozen; forward_train's eval path does not mutate its
    # foreground normalizer. The loss graph must remain active for dL/dF.
    feature_for_gradient=target.detach().requires_grad_(True)
    teacher_loss=model.teacher.loss(feature_for_gradient,masks,data['metas'],data['gt_segments'],data['gt_labels'])['cost']
    gradient=torch.autograd.grad(teacher_loss,feature_for_gradient)[0].detach().float()
    clip_valid=valid.reshape(batch,48,8).any(-1)
    proxy_values=[]
    for prediction in (prediction0,prediction8):
        proxy_values.append((gradient*(target.float()-prediction.detach().float())).abs().reshape(batch,-1,48,8).sum((1,3)))
    losses['utility']=.1*sum(proxy_loss(utility_logits[...,i],value,clip_valid) for i,value in enumerate(proxy_values))

    h,w=capture['spatial_shape'];spatial_loss=prediction0.sum()*0;score_loss=spatial_loss
    token_weights=weights.reshape(batch*48,8,1).expand(-1,-1,h*w).flatten(1)
    token_valid=valid.reshape(batch*48,8)
    for index in range(8,12):
        inputs=capture['mlp_inputs'][index];expected=capture['mlp_outputs'][index]
        predicted=model.aux.surrogates[str(index)](inputs)
        error=F.smooth_l1_loss(predicted.float(),expected.float(),reduction='none').mean(-1)
        spatial_loss=spatial_loss+(error*token_weights).sum()/token_weights.sum().clamp_min(1)
        magnitude=(predicted.detach().float()-expected.float()).abs().mean(-1).reshape(-1,8,h*w)
        scores=model.aux.spatial_scores[str(index)](inputs).squeeze(-1).reshape(-1,8,h*w)
        score_loss=score_loss+proxy_loss(scores.reshape(-1,h*w),magnitude.reshape(-1,h*w),
                                      token_valid.flatten()[:,None].expand(-1,h*w))
    losses['spatial_feature']=.25*spatial_loss
    losses['spatial_score']=.025*score_loss
    losses['cost']=sum(losses.values())
    if capture['attention_clips'] != [batch*48]*12 or capture['mlp_tokens'] != [batch*48*8*h*w]*12:
        raise RuntimeError('D1 teacher must execute every physical clip/token/layer')
    diagnostics=dict(regime='D1_dense_forward',teacher_clips=capture['dense_clips'],teacher_layers=capture['dense_layers'],
                     teacher_attention_tokens_per_layer=batch*48*8*h*w,teacher_heavy_mlp_tokens_per_layer=batch*48*8*h*w,
                     measured_attention_clips=capture['attention_clips'],measured_mlp_tokens=capture['mlp_tokens'],
                     teacher_gradient_l1=float(gradient.abs().mean()),
                     crop_created_endpoints=sum(int((~x).sum()) for x in data['gt_boundary_validity']))
    return losses,diagnostics
