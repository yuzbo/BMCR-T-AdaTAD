"""Reference mechanisms for FPW-TAD. No project checkpoints are used.

Conventions:
  anchors [B,K/2,C] are features of selected *pairs*, not a subset of
  original-time dense features. Decoder output is original native [B,Q,C].
  Invalid padding has an explicit mask. The caller supplies physical time.
  These modules are prototypes; production integration is an agent task.
"""
from dataclasses import dataclass
import math
from typing import Dict
import torch
from torch import Tensor, nn
import torch.nn.functional as F

@dataclass
class AnchorProvenance:
    source_indices: Tensor       # [B,K/2,2]; padded contributors remain flagged
    source_times: Tensor         # [B,K/2,2]
    contributor_valid: Tensor    # [B,K/2,2]
    center: Tensor               # [B,K/2], mean of valid contributors only
    span: Tensor                 # [B,K/2], difference if both contributors valid
    valid: Tensor                # [B,K/2]
    valid_fraction: Tensor       # [B,K/2]


def anchor_provenance(indices: Tensor, selected_valid: Tensor,
                      frame_times: Tensor) -> AnchorProvenance:
    if indices.ndim != 2 or indices.shape != selected_valid.shape:
        raise ValueError('indices/selected_valid must be matching [B,K]')
    if indices.dtype != torch.long or selected_valid.dtype != torch.bool:
        raise TypeError('indices must be long, selected_valid must be bool')
    b, k = indices.shape
    if k == 0 or k % 2 or frame_times.ndim != 2 or len(frame_times) != b:
        raise ValueError('K must be positive/even and frame_times must be [B,T]')
    if (indices < 0).any() or (indices >= frame_times.shape[1]).any():
        raise ValueError('indices outside physical storage')
    if not selected_valid.any(-1).all():
        raise ValueError('each sample must have at least one valid observation')
    # Prefix validity mirrors the pinned project, not arbitrary holes in storage.
    counts = selected_valid.sum(-1)
    expected = torch.arange(k, device=indices.device)[None] < counts[:, None]
    if not torch.equal(expected, selected_valid):
        raise ValueError('valid selected observations must be a prefix')
    for row in range(b):
        kept = indices[row, selected_valid[row]]
        if len(kept) > 1 and not (kept.diff() > 0).all():
            raise ValueError('valid selected indices must be strictly increasing')
    times = frame_times.gather(1, indices).reshape(b, k // 2, 2)
    ids = indices.reshape(b, k // 2, 2)
    flags = selected_valid.reshape(b, k // 2, 2)
    count = flags.sum(-1)
    center = (times * flags).sum(-1) / count.clamp_min(1)
    span = torch.where(count == 2, times[..., 1] - times[..., 0], 0.)
    return AnchorProvenance(ids, times, flags, center, span,
                            count > 0, count.to(times.dtype) / 2)


def interpolate_anchors(features: Tensor, centers: Tensor, valid: Tensor,
                        queries: Tensor) -> Tensor:
    """Constant edge extension; gradients flow to features, not coordinates.

    This is a geometric baseline, NOT an inverse of VideoMAE contextual mixing.
    """
    if features.ndim != 3 or centers.shape != features.shape[:2] or valid.shape != centers.shape:
        raise ValueError('features [B,A,C], centers/valid [B,A] required')
    if queries.ndim != 2 or len(queries) != len(features):
        raise ValueError('queries must be [B,Q]')
    rows = []
    for f, t, mask, q in zip(features, centers.detach(), valid, queries.detach()):
        f, t = f[mask], t[mask]
        if not len(t):
            raise ValueError('no valid anchors')
        if len(t) == 1:
            rows.append(f[:1].expand(len(q), -1)); continue
        if not bool((t.diff() > 0).all()):
            raise ValueError('valid centers must be strictly increasing')
        right = torch.searchsorted(t.contiguous(), q.contiguous()).clamp(1, len(t)-1)
        left = right - 1
        w = ((q-t[left])/(t[right]-t[left])).clamp(0, 1)
        rows.append(f[left]*(1-w[:, None]) + f[right]*w[:, None])
    return torch.stack(rows)


class QueryBlock(nn.Module):
    """Independent output queries: cross-attention + tokenwise FFN, no self-attention."""
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.norm_q = nn.LayerNorm(width)
        self.norm_m = nn.LayerNorm(width)
        self.cross = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.norm_ffn = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 4*width), nn.GELU(), nn.Linear(4*width, width))

    def forward(self, query: Tensor, memory: Tensor, valid: Tensor) -> Tensor:
        m = self.norm_m(memory)
        update, _ = self.cross(self.norm_q(query), m, m, key_padding_mask=~valid,
                               need_weights=False)
        query = query + update
        return query + self.ffn(self.norm_ffn(query))


class FullAxisDecoder(nn.Module):
    """Geometry/context-aware latent residual. Random initialization only.

    Official VideoMAE self-attention weights cannot be called a strict match to
    this cross-attention architecture. Pretrained-vs-random tests must first
    hold architecture constant and document every parameter conversion.
    """
    def __init__(self, channels: int, context_dim: int, width: int = 192,
                 heads: int = 3, layers: int = 2,
                 query_meta_dim: int = 6, anchor_meta_dim: int = 4):
        super().__init__()
        if width % heads or min(channels, width, layers) < 1:
            raise ValueError('invalid dimensions')
        self.memory_proj = nn.Linear(channels, width)
        self.base_proj = nn.Linear(channels, width)
        self.context_proj = nn.Linear(context_dim, width)
        self.query_meta = nn.Linear(query_meta_dim, width)
        self.anchor_meta = nn.Linear(anchor_meta_dim, width)
        self.query_token = nn.Parameter(torch.zeros(1, 1, width))
        self.blocks = nn.ModuleList([QueryBlock(width, heads) for _ in range(layers)])
        self.head = nn.Linear(width, channels)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, anchors: Tensor, anchor_valid: Tensor, baseline: Tensor,
                context: Tensor, query_meta: Tensor, anchor_meta: Tensor,
                query_valid: Tensor) -> Tensor:
        if not anchor_valid.any(-1).all():
            raise ValueError('at least one valid anchor is required per sample')
        if baseline.shape[:2] != query_valid.shape:
            raise ValueError('query shape mismatch')
        memory = self.memory_proj(anchors) + self.anchor_meta(anchor_meta)
        q = (self.base_proj(baseline) + self.context_proj(context)
             + self.query_meta(query_meta) + self.query_token)
        for block in self.blocks:
            q = block(q, memory, anchor_valid)
        # No fictitious observed-position overwrite: anchor and query grids differ.
        return (baseline + self.head(q)) * query_valid[..., None]


class PackedQueryAttention(nn.Module):
    """Actual selected Q / full K,V projections for ONE same-input clip.

    Input [B,N,C]; selected [B,M] unique indices. Returns residual on M rows.
    The production engine must scatter and then apply TIA on the full raster.
    """
    def __init__(self, channels: int, heads: int):
        super().__init__()
        if channels % heads:
            raise ValueError('head/channel mismatch')
        self.channels, self.heads = channels, heads
        self.qkv = nn.Linear(channels, 3*channels, bias=True)
        self.proj = nn.Linear(channels, channels)

    def _heads(self, value: Tensor) -> Tensor:
        b, n, c = value.shape
        return value.reshape(b, n, self.heads, c//self.heads).transpose(1, 2)

    def dense(self, x: Tensor) -> Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        y = F.scaled_dot_product_attention(self._heads(q), self._heads(k), self._heads(v), dropout_p=0.)
        return self.proj(y.transpose(1, 2).reshape_as(x))

    def forward(self, x: Tensor, selected: Tensor) -> Tensor:
        b, n, c = x.shape
        if selected.ndim != 2 or len(selected) != b:
            raise ValueError('selected must be [B,M]')
        if selected.numel() == 0:
            return x[:, :0]
        for ids in selected:
            if (ids < 0).any() or (ids >= n).any() or ids.unique().numel() != len(ids):
                raise ValueError('invalid selected query ids')
        xq = x.gather(1, selected[..., None].expand(-1, -1, c))
        w, bias = self.qkv.weight, self.qkv.bias
        q = F.linear(xq, w[:c], bias[:c])
        k = F.linear(x, w[c:2*c], bias[c:2*c])
        v = F.linear(x, w[2*c:], bias[2*c:])
        y = F.scaled_dot_product_attention(self._heads(q), self._heads(k), self._heads(v), dropout_p=0.)
        return self.proj(y.transpose(1, 2).reshape(b, selected.shape[1], c))


def layer_macs(n: int, c: int, queries: int, heavy_mlp: int) -> Dict[str, int]:
    """Matmul MAC only. TIA, light path, decoder, routing, kernels excluded."""
    if not (0 <= queries <= n and 0 <= heavy_mlp <= n):
        raise ValueError('counts outside raster')
    return {'kv_projection': 2*n*c*c if queries > 0 else 0,
            'q_and_output_projection': 2*queries*c*c,
            'qk_av': 2*queries*n*c,
            'heavy_mlp': 8*heavy_mlp*c*c}


def dense_reference_macs(c: int) -> Dict[str, int]:
    """Pinned 160x160, 768 candidates, 12-layer S/B configuration estimate."""
    n, clips, layers, d = 800, 48, 12, c//4
    per_layer = layer_macs(n, c, n, n)
    result = {k: v*clips*layers for k, v in per_layer.items()}
    result['patch_embedding'] = clips*n*(3*2*16*16)*c
    result['tia'] = clips*layers*n*(2*c*d+d*d+3*d)
    if c == 384: result['detector'] = 14456254464
    elif c == 768: result['detector'] = 14909239296
    else: raise ValueError('detector reference available only for S/B')
    return result
