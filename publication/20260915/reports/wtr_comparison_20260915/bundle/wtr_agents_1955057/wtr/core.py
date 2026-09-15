"""Small, testable primitives. Raw signed task gains are not information-theoretic values."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
import torch
from torch import Tensor, nn


def jsonable(x: Any) -> Any:
    if isinstance(x, Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if hasattr(x, 'tolist'):
        return x.tolist()
    return x


def digest(x: Any) -> str:
    return hashlib.sha256(json.dumps(jsonable(x), sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def tensor_digest(x: Tensor) -> str:
    x = x.detach().contiguous().cpu()
    h = hashlib.sha256(str((tuple(x.shape), str(x.dtype))).encode())
    h.update(memoryview(x.view(torch.uint8).numpy()))
    return h.hexdigest()


def file_digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2,
                              allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def read_jsonl(path: str | Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [r['action_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate action_id in {path}')
    return rows


def align_records(*collections: list[dict]) -> list[tuple[dict, ...]]:
    """Exact alignment, never an inner join that silently discards inconvenient actions."""
    maps = [{r['action_id']: r for r in rows} for rows in collections]
    if not maps or any(set(m) != set(maps[0]) for m in maps):
        raise ValueError('The action universes differ; regenerate the same bank.')
    result = []
    for key in sorted(maps[0]):
        rows = tuple(m[key] for m in maps)
        for field in ('bank_hash', 'window_hash', 'support_id', 'split', 'action_type'):
            if len({r[field] for r in rows}) != 1:
                raise ValueError(f'Misaligned {field}: {key}')
        result.append(rows)
    return result


@torch.no_grad()
def extrapolate(anchor: Tensor, current: Tensor, beta: float,
                valid: Tensor | None = None, max_step: float | None = None) -> Tensor:
    """Extrapolate RAW task gains. beta=1 means CURRENT, not EMA.

    max_step, if used, is a fixed TRAIN-calibrated clipping limit in raw units.
    This is a safety constraint, not a proof of a better future teacher.
    """
    if anchor.shape != current.shape or not 1 <= beta <= 2:
        raise ValueError('Same shapes and conservative beta in [1,2] required.')
    if not torch.isfinite(anchor).all() or not torch.isfinite(current).all():
        raise ValueError('Non-finite value teacher.')
    step = (beta - 1.) * (current.float() - anchor.float())
    if max_step is not None:
        if max_step <= 0:
            raise ValueError('max_step must be positive.')
        step = step.clamp(-max_step, max_step)
    result = current.float() + step
    if valid is not None:
        if valid.shape != result.shape[:-1]:
            raise ValueError('valid mask must match the action axes.')
        result = torch.where(valid[..., None], result, torch.zeros_like(result))
    return result.detach()


def action_distribution(raw_values: Tensor, scale: Tensor, valid: Tensor | None = None,
                        temperature: float = 1.) -> Tensor:
    """Last axis=cls/reg components; penultimate=actions. Appends a zero-value STOP.

    All teachers and students must use the same fixed component scale, mask and T.
    STOP keeps an all-negative action set from forcing a harmful intervention.
    """
    if raw_values.ndim < 2 or temperature <= 0:
        raise ValueError('Expected [..., actions, components] and positive temperature.')
    if scale.shape != raw_values.shape[-1:] or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Invalid fixed component scales.')
    scores = (raw_values.float() / scale.to(raw_values)).mean(-1) / temperature
    if valid is not None:
        if valid.shape != scores.shape:
            raise ValueError('Mask shape mismatch.')
        scores = scores.masked_fill(~valid.bool(), -torch.inf)
    scores = torch.cat((scores, scores.new_zeros((*scores.shape[:-1], 1))), -1)
    return scores.softmax(-1)


def js_distill(student: Tensor, teacher: Tensor) -> Tensor:
    if student.shape != teacher.shape:
        raise ValueError('Probability support mismatch.')
    p = student.float(); q = teacher.detach().float()
    if (p < 0).any() or (q < 0).any():
        raise ValueError('Negative probability.')
    m = (p + q) / 2
    def kl(a: Tensor, b: Tensor) -> Tensor:
        return (a * (a.clamp_min(1e-12).log() - b.clamp_min(1e-12).log())).sum(-1)
    return (0.5 * kl(p, m) + 0.5 * kl(q, m)).mean()


def gain_per_cost(gain: Tensor, delta_cost: Tensor) -> Tensor:
    if (delta_cost <= 0).any():
        raise ValueError('Gain/cost only applies to positive-cost additions, not equal-cost swaps.')
    return gain / delta_cost


def exact_capacity(scores: Tensor, valid: Tensor, count: int,
                   groups: Tensor | None = None, min_per_group: int = 0) -> Tensor:
    """One-dimensional equal-cost capacity with optional coverage quotas.

    This is not a general knapsack or a submodular optimization guarantee.
    """
    if scores.ndim != 1 or scores.shape != valid.shape:
        raise ValueError('Expected one-dimensional scores and valid mask.')
    if count < 0 or count > int(valid.sum()) or min_per_group < 0:
        raise ValueError('Infeasible capacity.')
    if not torch.isfinite(scores[valid]).all():
        raise ValueError('Valid scores must be finite.')
    chosen = torch.zeros_like(valid, dtype=torch.bool)
    if groups is not None:
        if groups.shape != scores.shape:
            raise ValueError('Group shape mismatch.')
        for group in torch.unique(groups[valid]):
            ids = ((groups == group) & valid).nonzero().flatten()
            if len(ids) < min_per_group:
                raise ValueError('Coverage quota cannot be met.')
            order = torch.argsort(scores[ids], descending=True, stable=True)
            chosen[ids[order[:min_per_group]]] = True
    remaining = count - int(chosen.sum())
    if remaining < 0:
        raise ValueError('Coverage quotas exceed capacity.')
    ids = (valid & ~chosen).nonzero().flatten()
    order = torch.argsort(scores[ids], descending=True, stable=True)
    chosen[ids[order[:remaining]]] = True
    return chosen


def taylor_residual_gain(gradient: Tensor, heavy: Tensor, cheap: Tensor) -> Tensor:
    """First-order ADD-heavy improvement at a cheap state, not an oracle label.

    Evaluating gradient at a dense state changes its meaning; record that origin.
    """
    if gradient.shape != heavy.shape or heavy.shape != cheap.shape:
        raise ValueError('Residual/gradient shape mismatch.')
    return -(gradient.detach().float() * (heavy.detach().float() - cheap.detach().float())).sum(-1)


class OperatorValueHead(nn.Module):
    """Proposed operator-specific scorer. Integration in engine remains an agent task.

    Features: pre-op state summary + physical support + normalized depth + age + budget.
    Outputs: signed cls/reg gain and uncertainty for attention / FFN replacement.
    """
    def __init__(self, feature_dim: int, num_operators: int = 2, width: int = 64):
        super().__init__()
        self.num_operators = num_operators
        self.net = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, width),
                                 nn.GELU(), nn.Linear(width, 4 * num_operators))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        x = self.net(features.float()).unflatten(-1, (self.num_operators, 4))
        return x[..., :2], x[..., 2:].clamp(-8, 8)
