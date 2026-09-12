"""Runtime matrix/conv counter shared by full profiles and actual interventions."""
def matrix_counter():
    from tools.full_eval import ArithmeticCounter
    class Counter(ArithmeticCounter):
        extra={'aten._scaled_dot_product_efficient_attention.default','aten._scaled_dot_product_attention_math.default'}
        counted=ArithmeticCounter.counted|extra
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            if str(func) not in self.extra:return super().__torch_dispatch__(func,types,args,kwargs)
            result=func(*args,**(kwargs or {}));self.operations[str(func)]+=1
            cost=self.attention_macs(*args[:3]);self.macs[self.phase[-1] if self.phase else 'routing_and_other']+=cost
            self.fused_attention.append(dict(q=list(args[0].shape),k=list(args[1].shape),v=list(args[2].shape),macs=cost));return result
    return Counter()

def counted_native_and_loss(model,data,selection):
    counter=matrix_counter()
    with counter:
        value,_=model.forward_native(data,selection=selection,apply_router=False)
        loss=model.teacher.loss(value,data)
    if counter.unresolved_matrix_ops():raise RuntimeError(counter.unresolved_matrix_ops())
    return value,loss,2*sum(counter.macs.values())
