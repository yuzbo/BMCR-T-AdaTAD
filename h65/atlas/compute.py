"""Input-preserving interventions with actual selective attention/FFN execution.

These are diagnostic actions on the official AdaTAD model, not a new trained
method. The original 384-position state and every global TIA adapter remain.
"""
from contextlib import contextmanager
import types
import torch
import torch.nn.functional as F


def selected_attention(attention, x, admitted):
    """Selected Q / full within-pack KV; inactive packs do not execute QKV."""
    b, n, c = x.shape
    out = torch.zeros_like(x)
    counts = admitted.sum(-1)
    for count in counts.unique().tolist():
        if not count:
            continue
        rows = (counts == count).nonzero().flatten()
        value = x[rows]
        if count == n:
            out[rows] = attention(value)
            continue
        indices = admitted[rows].nonzero()[:, 1].reshape(len(rows), count)
        query = value.gather(1, indices[..., None].expand(-1, -1, c))
        qb = getattr(attention, 'q_bias', None)
        vb = getattr(attention, 'v_bias', None)
        kvb = torch.cat((torch.zeros_like(vb), vb)) if vb is not None else None
        q = F.linear(query, attention.qkv.weight[:c], qb)
        kv = F.linear(value, attention.qkv.weight[c:], kvb)
        heads = attention.num_heads
        q = q.reshape(len(rows), count, heads, -1).transpose(1, 2)
        kv = kv.reshape(len(rows), n, 2, heads, -1).permute(2, 0, 3, 1, 4)
        delta = F.scaled_dot_product_attention(q, kv[0], kv[1], dropout_p=0.)
        delta = attention.proj_drop(attention.proj(delta.transpose(1, 2).reshape(len(rows), count, c)))
        row_result = torch.zeros_like(value)
        row_result.scatter_(1, indices[..., None].expand(-1, -1, c), delta)
        out[rows] = row_result
    return out


def routed_block(block, x, h, w, attention_mask, ffn_mask):
    # Norms are applied to the resident state; expensive matrix operations only
    # execute on the admitted positions. There is no full-then-zero FFN path.
    x = x + block.drop_path(selected_attention(block.attn, block.norm1(x), attention_mask))
    delta = torch.zeros_like(x)
    if bool(ffn_mask.any()):
        delta[ffn_mask] = block.mlp(block.norm2(x)[ffn_mask])
    x = x + block.drop_path(delta)
    return block.adapter(x, h, w) if block.use_adapter else x


class DenseInterventions:
    def __init__(self, vit):
        self.vit = vit
        self.depth = len(vit.blocks)
        self.attention = None
        self.ffn = None

    @contextmanager
    def apply(self, attention=None, ffn=None):
        originals = []
        if attention is None:
            yield
            return
        for layer, block in enumerate(self.vit.blocks):
            original = block.forward
            originals.append((block, original))
            def forward(this, x, h, w, layer=layer, original=original):
                am = attention[layer].reshape(x.shape[:2]).to(x.device)
                fm = ffn[layer].reshape(x.shape[:2]).to(x.device)
                if bool(am.all()) and bool(fm.all()):
                    return original(x, h, w)
                return routed_block(this, x, h, w, am, fm)
            block.forward = types.MethodType(forward, block)
        try:
            yield
        finally:
            for block, original in originals:
                block.forward = original


def masks(depth=12, native_time=384, height=10, width=10, full=False,
          valid_native=None, device='cpu'):
    am = torch.full((depth, native_time, height, width), full, dtype=torch.bool, device=device)
    fm = am.clone()
    if not full:
        am[0] = am[-1] = True
        fm[0] = fm[-1] = True
        # Padding receives the same dense work in every allocation policy; an
        # oracle cannot gain an advantage merely by discovering padding.
        if valid_native is not None:
            am[:, ~valid_native] = True
            fm[:, ~valid_native] = True
    return am, fm


def apply_action(am, fm, action, enabled=True):
    kind = action['axis']
    if kind == 'T':
        t0, t1 = action['native_range']
        layers = action.get('layers', list(range(1, len(am)-1)))
        for layer in layers:
            am[layer, t0:t1] = enabled
            fm[layer, t0:t1] = enabled
    elif kind == 'S':
        y0, y1, x0, x1 = action['patch_box']
        t0,t1 = action.get('native_range',(0,am.shape[1]))
        layers = action.get('layers', list(range(1, len(am)-1)))
        for layer in layers:
            fm[layer, t0:t1, y0:y1, x0:x1] = enabled
    elif kind == 'D':
        t0,t1 = action.get('native_range',(0,am.shape[1]))
        am[action['layer'],t0:t1] = enabled
        fm[action['layer'],t0:t1] = enabled
    else:
        raise ValueError(kind)
    return am, fm


def observation_intervention(inputs, start, stop, kind):
    """RGB values change only in [start, stop); shape and timestamps are fixed."""
    out = inputs.clone(memory_format=torch.contiguous_format)
    shape = out.shape
    x = out.reshape(-1, *shape[-4:])  # [batch*crops,3,T,H,W]
    t = x.shape[2]
    left = max(0, start-1)
    right = min(t-1, stop)
    if kind == 'neighbor':
        source = left if start > 0 else right
        x[:, :, start:stop] = x[:, :, source:source+1].expand(-1, -1, stop-start, -1, -1)
    elif kind == 'interpolate':
        fraction = torch.linspace(0, 1, stop-start+2, device=x.device, dtype=x.dtype)[1:-1]
        a, b = x[:, :, left:left+1].clone(), x[:, :, right:right+1].clone()
        x[:, :, start:stop] = a*(1-fraction[None,None,:,None,None])+b*fraction[None,None,:,None,None]
    elif kind == 'low_resolution':
        block = x[:, :, start:stop].permute(0, 2, 1, 3, 4)
        flat = block.reshape(-1, 3, shape[-2], shape[-1])
        small = F.interpolate(flat, size=(40, 40), mode='area')
        large = F.interpolate(small, size=shape[-2:], mode='bilinear', align_corners=False)
        x[:, :, start:stop] = large.reshape_as(block).permute(0, 2, 1, 3, 4)
    elif kind == 'hard_mask':
        x[:, :, start:stop] = 0
    else:
        raise ValueError(kind)
    return out


def execution_shape_key(am, fm):
    """Explicit executed shapes, used to reuse expensive operator profiles.

    Different positions with the same per-pack query counts execute the same
    matrix shapes; no content-based branch or theoretical keep-ratio is used.
    """
    p = am.shape[-1]*am.shape[-2]
    aq = am.reshape(len(am), -1, 8*p).sum(-1)
    fq = fm.reshape(len(fm), -1).sum(-1)
    return tuple(tuple(sorted(x.tolist())) for x in aq), tuple(fq.tolist())
