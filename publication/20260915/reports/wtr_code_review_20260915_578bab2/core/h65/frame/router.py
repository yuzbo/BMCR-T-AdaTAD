"""Bounded candidate-frame exchanges scored by signed actual-action supervision."""
import torch
from torch import nn
from h65.transport import Selection


def swap_selection(selection,row,remove,insert):
    values={k:getattr(selection,k).clone() for k in Selection.__dataclass_fields__}
    valid=values['valid'][row];ids=values['indices'][row,valid]
    if not bool((ids==remove).any()) or bool((ids==insert).any()):raise ValueError('Swap must replace one member by one nonmember')
    changed=torch.cat((ids[ids!=remove],ids.new_tensor([insert]))).sort().values
    values['indices'][row,valid]=changed;values['continuous'][row,valid]=changed.float()
    return Selection(**values)


def candidate_pairs(output,selection,masks,scope='local'):
    member=torch.zeros_like(masks,dtype=torch.long).scatter_add(1,selection.indices,selection.valid.long())>0
    pairs=[]
    for row in range(len(masks)):
        selected=member[row].nonzero().flatten();candidates=(masks[row]&~member[row]).nonzero().flatten()
        for insert in candidates.tolist():
            eligible=selected[(selected//16)==insert//16] if scope=='local' else selected
            if len(eligible):
                remove=eligible[(eligible-insert).abs().argmin()]
                pairs.append((row,int(remove),insert))
    return pairs


class ActionRouter(nn.Module):
    def __init__(self):
        super().__init__();self.network=nn.Sequential(nn.LayerNorm(294),nn.Linear(294,64),nn.GELU(),nn.Linear(64,2))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
        self.register_buffer('scales',torch.tensor([.001,.001]))

    def features(self,output,selection,masks,pairs):
        hidden=output['hidden'].detach();b,t,c=hidden.shape
        member=torch.zeros_like(masks,dtype=torch.long).scatter_add(1,selection.indices,selection.valid.long())>0
        mean=(hidden*member[...,None]).sum(1)/member.sum(-1,keepdim=True).clamp_min(1)
        rows=[]
        for row,remove,insert in pairs:
            scalars=hidden.new_tensor([remove/t,insert/t,(insert-remove)/t,
                float(output['action_logits'][row,remove].sigmoid()),float(output['action_logits'][row,insert].sigmoid()),
                float(selection.valid[row].sum()/masks[row].sum())])
            rows.append(torch.cat((hidden[row,remove],hidden[row,insert],mean[row],scalars)))
        return torch.stack(rows) if rows else hidden.new_empty((0,294))

    def predict(self,output,selection,masks,pairs):return self.network(self.features(output,selection,masks,pairs).float())

    def refine(self,output,selection,masks,scope='local'):
        pairs=candidate_pairs(output,selection,masks,scope)
        if not pairs:return selection,dict(changes=[],predicted_gain_sum=0.)
        values=self.predict(output,selection,masks,pairs);score=values.mean(-1);result=selection;changes=[]
        for row in range(len(masks)):
            valid=[i for i,p in enumerate(pairs) if p[0]==row]
            if not valid:continue
            indices=score.new_tensor(valid,dtype=torch.long);best=indices[score[indices].argmax()]
            if float(score[best])>0:
                _,remove,insert=pairs[int(best)];result=swap_selection(result,row,remove,insert)
                changes.append(dict(row=row,remove=remove,insert=insert,predicted=float(score[best])))
        return result,dict(changes=changes,predicted_gain_sum=sum(x['predicted'] for x in changes))

    @torch.no_grad()
    def update_scales(self,targets):
        self.scales.mul_(.9).add_(targets.detach().abs().mean(0).clamp_min(1e-4),alpha=.1)
