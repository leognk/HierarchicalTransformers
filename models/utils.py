import torch
import einops
import einops.layers.torch
import math
import re
import itertools
import utils


#################### UTILITY ####################


def flat_dim(dim, size=None):
    return dim * utils.product(size) if size else dim


def geometric_mean(it):
    return utils.product(it) ** (1 / len(it))


def prime_factors(n):
    res = []
    i = 2
    while i * i <= n:
        while n % i == 0:
            res.append(i)
            n //= i
        i += 1
    if n > 1:
        res.append(n)
    return res


def decompose_patching(patch_size):
    """Decompose patching in patch_size into a succession of irreducible patching."""
    # Each patching i scales the input on axis n by patch_sizes[i][n].
    # So the total scaling factor on axis n is the product of patch_sizes[i][n], 0 <= i < n_patches,
    # which equals patch_size[n] because patch_sizes[i][n] are its prime factors.
    patch_size_decomposed = [prime_factors(n) for n in patch_size]
    n_patches = max(len(dec) for dec in patch_size_decomposed)
    patch_size_decomposed = [dec + ([1] * (n_patches - len(dec))) for dec in patch_size_decomposed]
    patch_sizes = list(zip(*patch_size_decomposed))
    return patch_sizes


#################### OPERATIONS ####################


def ceil_div_abs(x, y):
    if utils.sign(x) == utils.sign(y):
        return utils.ceil_div(x, y)
    return int(math.floor(round(x / y, 8))) # round to eliminate precision errors


def hardmax(logits, dim):
    y_soft = logits.softmax(dim)
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
    res = y_hard - y_soft.detach() + y_soft
    return res


#################### TWO-WAY LIST ####################


class TwoWayList:
    """
    A default list that can be extended in both right and left sides.
    The right side is indexed from 0 to (nr - 1) and the left side from -1 to -nl.
    The list is extended when trying to access indices nr or (-nl - 1).
    Trying to access indices beyond that raises an error.
    """

    def __init__(self, default_factory):
        self.default_factory = default_factory
        self.right, self.left = [], []
    
    @property
    def start_idx(self):
        return -len(self.left)
    
    @property
    def stop_idx(self):
        return len(self.right)
    
    def __iter__(self):
        return itertools.chain(reversed(self.left), self.right)
    
    def verify_index(self, i):
        if i >= 0:
            if i > len(self.right):
                raise IndexError(f"Index {i} is out of range")
            elif i == len(self.right):
                self.right.append(self.default_factory())
        elif i < 0:
            if -i > len(self.left) + 1:
                raise IndexError(f"Index {i} is out of range")
            elif -i == len(self.left) + 1:
                self.left.append(self.default_factory())
        else:
            raise IndexError("Index 0 is forbidden")
    
    def __getitem__(self, i):
        self.verify_index(i)
        if i >= 0: return self.right[i]
        else: return self.left[-i - 1]
    
    def __setitem__(self, i, value):
        self.verify_index(i)
        if i >= 0: self.right[i] = value
        else: self.left[-i - 1] = value
    
    def __repr__(self):
        res = ""
        nl, nr = len(self.left), len(self.right)
        for i, x in enumerate(self.left[::-1]):
            res += f"{i - nl}: {x}\n"
        for i, x in enumerate(self.right):
            res += f"{i}: {x}"
            if i != nr - 1: res += "\n"
        return res


#################### PADDING ####################


def get_missing_pad(in_size, group_size):
    """
    Return the padding needed to make in_size divisible by group_size.
    in_size: [N1 ... Nd]
    group_size: [n1 ... nd]
    output shape: [p1 ... pd] where pi = ni - (Ni % ni) if Ni % ni != 0 else 0
    """
    d = len(in_size)
    need_pad = False
    pad = [0] * d
    for i in range(d):
        r = in_size[i] % group_size[i]
        if r != 0:
            need_pad = True
            pad[i] = group_size[i] - r
    return need_pad, pad


def get_real_missing_pad(x_size, real_size, group_size):
    """
    Same as get_missing_pad but the padding is calculated from the theoretical
    real continuous size of the unpadded input to prevent rounding issues.
    """
    d = len(x_size)
    need_pad = False
    pad = [0] * d
    for i in range(d):
        n_groups = utils.ceil_div(real_size[i], group_size[i])
        padded_size = n_groups * group_size[i]
        p = padded_size - x_size[i]
        if p != 0:
            need_pad = True
            pad[i] = p
    return need_pad, pad


def to_torch_pad(pd, emb_dim_last=None):
    """
    Convert pad to the format required by torch.nn.functional.pad.
    pad: [p1 ... pd]
    torch pad:
        if emb_dim_last is None: [0 pd ... 0 p3 0 p2 0 p1]
        elif emb_dim_last: [0 0 0 pd ... 0 p3 0 p2 0 p1]
        else: [0 pd ... 0 p3 0 p2 0 p1 0 0]
    """
    need_pad, pad = pd
    if not need_pad: return pd
    d = len(pad)
    torch_pad = [0] * (2 * (d + 1))
    for i in range(d):
        j = -2 * i - 1
        if not emb_dim_last: j -= 2
        torch_pad[j] = pad[i]
    if emb_dim_last is None: torch_pad = torch_pad[:-2]
    return need_pad, tuple(torch_pad)


def do_pad(x, pd, value=0):
    """Apply padding to x with pd = (need_pad, pad)."""
    need_pad, pad = pd
    if need_pad:
        x = torch.nn.functional.pad(x, pad, mode='constant', value=value)
    return x


def inv_pad(pd):
    """Return the reverse operation of pd = (need_pad, pad)."""
    need_pad, pad = pd
    if need_pad:
        pad = tuple(-p for p in pad)
    return need_pad, pad


def pad_missing(x, group_size, emb_dim_last):
    """
    Pad with the needed amount to make x's size divisible by group_size.
    x shape: [b n1 ... nd d] if emb_dim_last else [b c n1 ... nd]
    """
    size = x.shape[1:-1] if emb_dim_last else x.shape[2:]
    pd = get_missing_pad(size, group_size)
    pd = to_torch_pad(pd, emb_dim_last)
    x = do_pad(x, pd)
    cp = inv_pad(pd)
    return x, cp


class Pad(torch.nn.Module):

    def __init__(self, pd):
        super().__init__()
        self.pd = pd
    
    def forward(self, x):
        return do_pad(x, self.pd)


#################### TORCH MODULES ####################


class Transpose(torch.nn.Module):

    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0, self.dim1 = dim0, dim1
    
    def forward(self, x):
        return torch.transpose(x, self.dim0, self.dim1)


class MoveDim(torch.nn.Module):

    def __init__(self, source, destination):
        super().__init__()
        self.source, self.destination = source, destination
    
    def forward(self, x):
        return torch.movedim(x, self.source, self.destination)


def get_nd_pattern(pattern, n_repeats, char_to_replace):
    def replacer(match):
        s = match.group(1)
        res = [s.replace(char_to_replace, str(i)) for i in range(1, n_repeats + 1)]
        return ' '.join(res)
    return re.sub(r'\[([^\]]+)\]', replacer, pattern)


def get_nd_axes_lengths(axes_lengths):
    res = {}
    for ax_prefix, ax_lengths in axes_lengths.items():
        res.update({f'{ax_prefix}{i + 1}': n for i, n in enumerate(ax_lengths)})
    return res


def get_einops_nd_args(pattern, axes_lengths, char_to_replace):
    n_repeats = len(next(iter(axes_lengths.values())))
    nd_pattern = get_nd_pattern(pattern, n_repeats, char_to_replace)
    nd_axes_lengths = get_nd_axes_lengths(axes_lengths)
    return nd_pattern, nd_axes_lengths


def rearrange_nd(tensor, pattern, axes_lengths, char_to_replace='0'):
    """
    Dynamic n-dimensional version of einops.rearrange.
    Lists in axes_lengths must be of the same length.
    axes_lenghts cannot be empty in order to infer the number of axes.
    Example:
    >>> rearrange_nd(
            t, 'b [(group0 block0 token0)] d -> b [group0] ([block0] [token0]) d',
            {'block': [2, 3], 'token': [5, 7]}
        )
    is equivalent to
    >>> einops.rearrange(
            t, 'b (group1 block1 token1) (group2 block2 token2) d -> b group1 group2 (block1 block2 token1 token2) d',
            block1=2, block2=3, token1=5, token2=7
        )
    """
    pattern, axes_lengths = get_einops_nd_args(pattern, axes_lengths, char_to_replace)
    return einops.rearrange(tensor, pattern, **axes_lengths)


class RearrangeNd(torch.nn.Module):

    def __init__(self, pattern, axes_lengths, char_to_replace='0'):
        super().__init__()
        pattern, axes_lengths = get_einops_nd_args(pattern, axes_lengths, char_to_replace)
        self.rearrange = einops.layers.torch.Rearrange(pattern, **axes_lengths)
    
    def forward(self, x):
        return self.rearrange(x)


def get_Conv(n):
    return {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}.get(n)


def get_ConvT(n):
    return {1: torch.nn.ConvTranspose1d, 2: torch.nn.ConvTranspose2d, 3: torch.nn.ConvTranspose3d}.get(n)


def get_BatchNorm(n):
    return {1: torch.nn.BatchNorm1d, 2: torch.nn.BatchNorm2d, 3: torch.nn.BatchNorm3d}.get(n)


class ConvNormActivation(torch.nn.Sequential):

    def __init__(self, n_axes, in_channels, out_channels, kernel_size, stride, padding, norm, activation, use_transpose=False):
        _conv = get_Conv if not use_transpose else get_ConvT
        conv = _conv(n_axes)
        layers = [conv(in_channels, out_channels, kernel_size, stride=stride, padding=padding), norm(out_channels), activation()]
        super().__init__(*layers)


class ConvBNReLU(ConvNormActivation):

    def __init__(self, n_axes, in_channels, out_channels, kernel_size, stride, padding=0):
        bn = get_BatchNorm(n_axes)
        super().__init__(n_axes, in_channels, out_channels, kernel_size, stride, padding, bn, torch.nn.ReLU)


class ConvTLNGELU(ConvNormActivation):

    def __init__(self, n_axes, in_channels, out_channels, kernel_size, stride, padding=0):
        norm = lambda n_channels: torch.nn.GroupNorm(num_groups=1, num_channels=n_channels)
        super().__init__(n_axes, in_channels, out_channels, kernel_size, stride, padding, norm, torch.nn.GELU, use_transpose=True)


#################### POSITIONAL EMBEDDINGS ####################


def get_nd_position_indices(axes_n_tokens, dtype=None, device=None):
    """
    Return n-dimensional position indices that can be turned into positional embeddings.
    axes_n_tokens: list of number of tokens for each axis.
    output shape: [d n1 ... nd] where ni = axes_n_tokens[i - 1] and d = n_axes.
    Example:
    >>> get_nd_position_indices([2, 3])
    torch.tensor([
        [[0, 0, 0],  # 1st axis
         [1, 1, 1]],

        [[0, 1, 2],  # 2nd axis
         [0, 1, 2]],
    ])
    """
    pos = [torch.arange(ax_n_tokens, dtype=dtype, device=device) for ax_n_tokens in axes_n_tokens]
    pos = torch.stack(torch.meshgrid(pos, indexing='ij'))
    return pos


def nd_pos_to_sin_posemb(pos, emb_dim, temperature=10000):
    """
    Return the sinusoidal positional embeddings of n-dimensional position indices pos.
    pos shape: [d n1 ... nd]
    output shape: [n1 ... nd emb_dim]
    Example:
    >>> pos = get_nd_position_indices([2, 3])
    >>> get_nd_sin_posemb(pos, 8)
    torch.tensor([
        [pe[0, 0], pe[0, 1], pe[0, 2]],
        [pe[1, 0], pe[1, 1], pe[1, 2]],
    ])
    where pe = [1st_axis_sin, 1st_axis_cos, 2nd_axis_sin, 2nd_axis_cos].
    """
    n_axes2 = pos.shape[0] * 2
    assert emb_dim > n_axes2; assert emb_dim % n_axes2 == 0
    slice_dim = emb_dim // n_axes2
    omega = torch.arange(slice_dim, dtype=pos.dtype, device=pos.device) / (slice_dim - 1)
    omega = 1.0 / temperature ** omega
    pos = pos.unsqueeze(-1) * omega # [d n1 ... nd 1] x [slice_dim] -> [d n1 ... nd slice_dim]
    posemb = torch.cat((pos.sin(), pos.cos()), dim=-1)
    posemb = torch.cat(tuple(posemb), dim=-1) # [d n1 ... nd (2 slice_dim)] -> [n1 ... nd (d 2 slice_dim)]
    return posemb


def get_nd_sin_posemb(axes_n_tokens, emb_dim, dtype=None, device=None, temperature=10000):
    return nd_pos_to_sin_posemb(get_nd_position_indices(axes_n_tokens, dtype, device), emb_dim, temperature)


#################### BACKBONE WITH HEAD ####################


class BackboneWithHead(torch.nn.Module):
    """
    Adds the methods necessary for layer-wise lr decay
    with an additional head if the backbone defines them.
    """

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
        if hasattr(self.backbone, "no_weight_decay"):
            self.no_weight_decay = self._no_weight_decay
        if hasattr(self.backbone, "num_lr_scale_groups"):
            self.num_lr_scale_groups = self._num_lr_scale_groups
        if hasattr(self.backbone, "lr_scale_group_id"):
            self.lr_scale_group_id = self._lr_scale_group_id
    
    @property
    def _no_weight_decay(self):
        nwd = self.backbone.no_weight_decay
        res = set()
        for n, _ in self.named_parameters():
            if any(x in n for x in nwd):
                res.add(n)
        return res
    
    @property
    def _num_lr_scale_groups(self):
        return self.backbone.num_lr_scale_groups + 1
    
    def _lr_scale_group_id(self, name):
        if name.startswith("backbone"):
            _name = name.split('.', 1)[1]
            return self.backbone.lr_scale_group_id(_name)
        elif name.startswith("head"):
            return self.num_lr_scale_groups - 1
        else:
            raise ValueError(f"Invalid parameter name: {name}")


#################### SANITY CHECK ####################


def disable_grads(model):
    rg = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad = False
    return rg


def restore_requires_grads(model, rg):
    for p, r in zip(model.parameters(), rg):
        p.requires_grads = r


def verify_no_batch_mix(model, batch_size, in_size):
    """
    Verify as a sanity check that the model does not mix information across the batch dimension
    by looking at the gradient of a dummy loss w.r.t. the input.
    Return True if the model passes the test.
    """
    rg = disable_grads(model)
    device = next(model.parameters()).device
    x = torch.randn(batch_size, *in_size, device=device, requires_grad=True)
    y = model(x)
    for i in range(batch_size):
        loss_i = y[i].sum()
        loss_i.backward(retain_graph=True)
        x_grad_no_i = torch.cat((x.grad[:i], x.grad[i+1:]))
        if not torch.all(x_grad_no_i == 0): return False
        x.grad = None
    restore_requires_grads(model, rg)
    return True