"""Two symbols the GRIN files import from their package root, reproduced here so
the author's model code runs unmodified."""
import torch

epsilon = 1e-8


def reverse_tensor(tensor=None, axis=-1):
    if tensor is None:
        return None
    if tensor.dim() <= 1:
        return tensor
    indices = range(tensor.size()[axis])[::-1]
    indices = torch.as_tensor(indices, dtype=torch.long, device=tensor.device)
    return tensor.index_select(axis, indices)
