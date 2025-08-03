import torch;
import torch.nn as nn;
from inspect import isfunction;

def expand_axis_like(a:torch.Tensor, b:torch.Tensor):
    """ Expands axes (at the end) of b to have the same number of axis as a.

    Args:
        a (Tensor): A reference tensor.
        b (Tensor): A target tensor to be expanded.

    Returns:
        Expanded version of b.

    """
    assert len(a.shape) >= len(b.shape), f"The number of axis in a must greater than b. Shape a: {a.shape}, Shape b: {b.shape}"
    n_unsqueeze = len(a.shape) - len(b.shape)
    b = b[(..., ) + (None, ) * n_unsqueeze] # unsqueeze such that it has the same size for broadcasting.
    return b

class Rescale(object):
    def __init__(self, old_range, new_range):
        self.old_range = old_range
        self.new_range = new_range

    def __call__(self, image):
        old_min, old_max = self.old_range
        new_min, new_max = self.new_range
        image -= old_min
        image *= (new_max - new_min) / (old_max - old_min)
        image += new_min
        return image

normalize_to_neg_one_to_one = Rescale((0, 1), (-1, 1));
unnormalize_to_zero_to_one = Rescale((-1, 1), (0, 1));


def make_layer(block, n_layers, seq=False):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    if seq:
        return nn.Sequential(*layers)
    else:
        return nn.ModuleList(layers)

def exists(x):
    return x is not None
    
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))