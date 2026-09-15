"""Detached selected-rank geometry, inverted before official window/NMS logic."""
import torch
from h65.transport import Selection


class TrueTimeMap:
    def __init__(self, positions, valid_length):
        true = positions.detach().float()
        if not len(true) or not bool((true.diff() > 0).all()):
            raise ValueError("positions must be nonempty and strictly increasing")
        rank = torch.arange(len(true), device=true.device, dtype=torch.float32)
        if true[0] > 0:
            true = torch.cat((true.new_zeros(1), true))
            rank = torch.cat((rank.new_full((1,), -1), rank))
        self.true = torch.cat((true, true.new_tensor([valid_length])))
        self.rank = torch.cat((rank, rank.new_tensor([len(positions)])))

    @staticmethod
    def interpolate(value, source, destination):
        shape = value.shape
        value = value.detach().float().reshape(-1).clamp(source[0], source[-1])
        right = torch.searchsorted(source.contiguous(), value.contiguous()).clamp(1, len(source) - 1)
        left = right - 1
        ratio = (value - source[left]) / (source[right] - source[left]).clamp_min(1e-6)
        return (destination[left] + ratio * (destination[right] - destination[left])).reshape(shape)

    def to_rank(self, value):
        return self.interpolate(value, self.true, self.rank)

    def to_true(self, value):
        return self.interpolate(value, self.rank, self.true)


def maps_for(selection, masks):
    return [TrueTimeMap(p[v], int(m.sum())) for p, v, m in zip(selection.indices, selection.valid, masks)]


def mix_rows(learned, uniform, teacher_rows):
    values = {}
    for key in Selection.__dataclass_fields__:
        a, b = getattr(learned, key), getattr(uniform, key)
        values[key] = torch.where(teacher_rows[:, None], b, a)
    return Selection(**values)


def occupancy(selection, masks):
    """Hard occupancy forward with the reference's local mass-transport gradient."""
    hard = selection.indices
    lengths = masks.sum(-1, keepdim=True)
    direction = torch.where(hard < lengths - 1, 1, -1)
    neighbor = (hard + direction).clamp_min(0)
    mass = (selection.continuous - selection.continuous.detach()) * direction * selection.valid
    result = selection.rates.new_zeros(masks.shape)
    result = result.scatter_add(1, hard, selection.valid.float() - mass)
    result = result.scatter_add(1, neighbor, mass)
    return result * masks


def exchange(selection, row, remove, insert):
    """One exact-cardinality swap; metadata and all other rows retained."""
    values = {key: getattr(selection, key).detach().clone() for key in Selection.__dataclass_fields__}
    n = int(selection.valid[row].sum())
    selected = values['indices'][row, :n]
    if int((selected == remove).sum()) != 1 or bool((selected == insert).any()):
        raise ValueError('exchange must remove a member and insert a nonmember')
    selected[selected == remove] = insert
    values['indices'][row, :n] = selected.sort().values
    values['continuous'][row] = values['indices'][row].float()
    return Selection(**values)
