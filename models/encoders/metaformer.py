from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from einops import rearrange, pack, unpack, repeat, reduce
from ..stems import create_stem
from ..utils import hardmax
from utils import ArgumentParser, get_args
import utils


class LayerNormGeneral(nn.Module):
    
    def __init__(self, affine_shape=None, normalized_dim=(-1,), scale=True, bias=True, eps=1e-5):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        self.weight = nn.Parameter(torch.ones(affine_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps

    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        if self.use_scale:
            x = x * self.weight
        if self.use_bias:
            x = x + self.bias
        return x


class LayerNormWithoutBias(nn.Module):
    """
    Equal to partial(LayerNormGeneral, bias=False) but faster, 
    because it directly utilizes otpimized F.layer_norm
    """
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.bias = None
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)


class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0, scale_learnable=True, bias_learnable=True, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1), requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1), requires_grad=bias_learnable)
        
    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class KernelPooling(nn.Module):
    
    def __init__(self, in_dim, out_dim, in_size, out_size, method, kernel_size, stride, padding, random=False):
        super().__init__()
        self.norm = LayerNormWithoutBias(in_dim, eps=1e-6)
        if method == 'conv':
            self.pool = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding)
            if random:
                for p in self.pool.parameters():
                    p.requires_grad = False
        elif method == 'avg':
            self.pool = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 1, 1, 0),
                nn.AvgPool2d(kernel_size, stride, padding),
            )
        elif method == 'max':
            self.pool = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 1, 1, 0),
                nn.MaxPool2d(kernel_size, stride, padding),
            )
        else:
            raise ValueError(f"Invalid method: {method}")

    def forward(self, x):
        """[b h w d] -> [b h' w' d']"""
        x = self.norm(x) # [b h w d]
        x = torch.movedim(x, -1, 1) # [b h w d] -> [b d h w]
        x = self.pool(x) # [b d h w] -> [b d' h' w']
        x = torch.movedim(x, 1, -1) # [b d' h' w'] -> [b h' w' d']
        return x


class SpatialMLPPooling(nn.Module):

    def __init__(self, in_dim, out_dim, in_size, out_size, use_mlp, random=False):
        super().__init__()
        n_in = utils.product(in_size)
        n_out = utils.product(out_size)
        self.h_out, self.w_out = out_size

        self.norm1 = LayerNormWithoutBias(in_dim, eps=1e-6)
        if use_mlp: self.spatial_proj = MLP(n_in, n_out / n_in, n_out)
        else: self.spatial_proj = nn.Linear(n_in, n_out, bias=False)
        if random:
            for p in self.spatial_proj.parameters():
                p.requires_grad = False
        self.norm2 = LayerNormWithoutBias(in_dim, eps=1e-6)
        self.channel_proj = nn.Linear(in_dim, out_dim, bias=False)
    
    def forward(self, x):
        """[b h w d] -> [b h' w' d']"""
        x = rearrange(x, 'b h w d -> b (h w) d') # [b h w d] -> [b n d]
        x = self.norm1(x) # [b n d]
        x = torch.movedim(x, -1, -2) # [b n d] -> [b d n]
        x = self.spatial_proj(x) # [b d n] -> [b d n']
        x = rearrange(x, 'b d (h w) -> b h w d', h=self.h_out, w=self.w_out) # [b d n'] -> [b h' w' d]
        x = self.norm2(x) # [b h' w' d]
        x = self.channel_proj(x) # [b h' w' d] -> [b h' w' d']
        return x


class Attention0Pooling(nn.Module):
    
    def __init__(self, in_dim, out_dim, in_size, out_size, clusters_init, assignment, last_assignment, res_scale_init_value=1.0):
        super().__init__()
        self.h_out, self.w_out = out_size
        self.clusters_init = clusters_init
        self.assignment = assignment

        self.norm1 = LayerNormWithoutBias(in_dim, eps=1e-6)
        self.attention = AttentionAggregate(in_dim, assignment=last_assignment)

        self.res_scale = Scale(dim=in_dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()
        
        self.norm2 = LayerNormWithoutBias(2 * in_dim, eps=1e-6)
        self.proj = nn.Linear(2 * in_dim, out_dim, bias=False)

    def forward(self, x, clusters):
        """
        x: [b h w d] -> [b h' w' d']
        clusters: [b (n' 2) d] ->
        token_sizes: -> [b h' w']
        """
        x, _ = pack([x], 'b * d') # [b h w d] -> [b n d]
        clusters2, token_sizes = self.attention(self.norm1(x), self.norm1(clusters))
        if token_sizes is not None:
            token_sizes = reduce(token_sizes, 'b (n two) -> b n', 'sum', two=2) # [b (n' 2)] -> [b n']

        x = self.res_scale(clusters) + clusters2 # [b n d] -> [b (n' 2) d]
        x = rearrange(x, 'b (n two) d -> b n (two d)', two=2) # [b (n' 2) d] -> [b n' (2 d)]
        x = self.proj(self.norm2(x)) # [b n' (2 d)] -> [b n' d']

        x = rearrange(x, 'b (h w) d -> b h w d', h=self.h_out, w=self.w_out) # [b n' d'] -> [b h' w' d]
        if token_sizes is not None:
            token_sizes = rearrange(token_sizes, 'b (h w) -> b h w', h=self.h_out, w=self.w_out) # [b n'] -> [b h' w']
        return x, token_sizes


class GreedyMerging(nn.Module):
    """
    Greedily merge similar tokens together.
    The number of reduced tokens is fixed.
    """

    def __init__(self, n_tokens, n_out):
        super().__init__()

        self.n = n_tokens
        self.r = n_tokens - n_out

        triu_mask = torch.triu(torch.ones(self.n, self.n, dtype=bool), diagonal=1) # [n n]
        self.register_buffer("triu_mask", triu_mask)

        sq_rge = torch.arange(self.n * self.n).view(self.n, self.n) # [n n]
        triu_ids = torch.cat([sq_rge[i][i + 1:] for i in range(self.n - 1)]) # [m]
        self.register_buffer("triu_ids", triu_ids)

        lin_rge = torch.arange(self.n)
        grid = torch.stack(torch.meshgrid(lin_rge, lin_rge, indexing='ij'), dim=-1) # [n n 2]
        triu_ijs = torch.cat([grid[i][i + 1:] for i in range(self.n - 1)]) # [m 2]
        self.register_buffer("triu_ijs", triu_ijs)
    
    def forward(self, k, v):
        """
        k: [b n d] ->
        v: [b n d] ->
        out: -> [b no d]
        group_sizes -> [b no]
        """
        b = k.shape[0]
        b_rge = torch.arange(b, device=k.device)

        # Compute scores.
        k = k / torch.norm(k, dim=-1, keepdim=True) # [b n d]
        scores = torch.matmul(k, k.transpose(-1, -2)) # [b n n]

        # Sort scores.
        scores = torch.gather(scores.view(b, -1), dim=1, index=self.triu_ids.expand(b, -1)) # [b m]
        sort_scores = torch.argsort(scores, dim=-1, descending=True) # [b m]

        # Sort the edges.
        sorted_ijs = torch.gather(
            self.triu_ijs.expand(b, -1, -1), dim=1, index=sort_scores.unsqueeze(-1).expand(-1, -1, 2)
        ) # [b m 2]
        sorted_masks = torch.ones(sort_scores.shape, dtype=bool, device=k.device) # [b m]

        # groups[b][i][j]: whether token j belongs to group i.
        groups = torch.eye(self.n, dtype=bool, device=k.device).repeat(b, 1, 1) # [b n n]
        # group_ids[b][i]: group idx of token i.
        group_ids = torch.arange(self.n, dtype=int, device=k.device).repeat(b, 1) # [b n]

        for _ in range(self.r):
            # Get edge (i, j).
            ij = sorted_ijs[b_rge, torch.max(sorted_masks, dim=1)[1]] # [b 2]
            i0, j0 = ij[:, 0], ij[:, 1] # [b], [b]
            i, j = group_ids[b_rge, i0], group_ids[b_rge, j0] # [b], [b]

            # Update groups & group_ids.
            group_i = groups[b_rge, i] # [b n]
            groups[b_rge, j] += group_i # [b n]
            group_ids[group_i] = torch.repeat_interleave(j, group_i.sum(1))
            groups[b_rge, i] = 0 # [b n]

            # Update sorted_masks.
            group_j = groups[b_rge, j] # [b n]
            edges_j = group_j.unsqueeze(1) * group_j.unsqueeze(2) # [b n n]
            edges_j = edges_j[self.triu_mask.expand(b, -1, -1)].view(b, -1) # [b m]
            edges_j = torch.gather(edges_j, dim=1, index=sort_scores) # [b m]
            sorted_masks *= torch.logical_not(edges_j) # [b m]

        # Remove null rows from groups.
        groups = groups[groups.sum(-1) != 0].view(b, -1, self.n) # [b no n]
        group_sizes = groups.sum(-1) # [b no]
        groups = groups / group_sizes[..., None] # [b no n]

        out = torch.matmul(groups, v) # [b no d]

        return out, group_sizes


class GeMePooling(nn.Module):
    
    def __init__(self, in_dim, out_dim, in_size, out_size):
        super().__init__()
        self.h_out, self.w_out = out_size
        n_in, n_out = utils.product(in_size), utils.product(out_size)

        self.norm1 = LayerNormWithoutBias(in_dim, eps=1e-6)
        self.to_kv = nn.Linear(in_dim, 2 * in_dim, bias=False)
        self.merge = GreedyMerging(n_in, n_out)
        
        self.norm2 = LayerNormWithoutBias(in_dim, eps=1e-6)
        self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        """
        x: [b h w d] -> [b h' w' d']
        token_sizes [b h' w'] ->
        """
        x, _ = pack([x], 'b * d') # [b h w d] -> [b n d]

        kv = self.to_kv(self.norm1(x))
        k, v = rearrange(kv, 'b n (k d) -> k b n d', k=2)
        x, token_sizes = self.merge(k, v) # [b n' d], [b n']
        x = self.proj(self.norm2(x)) # [b n' d] -> [b n' d']

        x = rearrange(x, 'b (h w) d -> b h w d', h=self.h_out, w=self.w_out) # [b n' d'] -> [b h' w' d]
        token_sizes = rearrange(token_sizes, 'b (h w) -> b h w', h=self.h_out, w=self.w_out) # [b n'] -> [b h' w']
        return x, token_sizes


def create_pooling(name, in_dim, out_dim, in_size, out_size, *args, **kwargs):
    model = {
        'conv': partial(KernelPooling, method='conv'),
        'avg': partial(KernelPooling, method='avg'),
        'max': partial(KernelPooling, method='max'),
        'sp_lin': partial(SpatialMLPPooling, use_mlp=False),
        'sp_mlp': partial(SpatialMLPPooling, use_mlp=True),
        'rand': partial(SpatialMLPPooling, use_mlp=False, random=True),
        'rand_conv': partial(KernelPooling, method='conv', random=True),
        'att0': Attention0Pooling,
        'geme': GeMePooling,
    }[name]
    pooling = model(in_dim, out_dim, in_size, out_size, *args, **kwargs)
    pooling.need_clusters = name == 'att0'
    pooling.output_token_sizes = name in {'att0', 'geme'}
    return pooling


class IdentityMixing(nn.Module):
    
    def __init__(self, size, emb_dim):
        super().__init__()

    def forward(self, x):
        return x


class RandomMixing(nn.Module):

    def __init__(self, size, emb_dim):
        super().__init__()
        n_tokens = utils.product(size)
        self.random_matrix = nn.parameter.Parameter(
            torch.softmax(torch.rand(n_tokens, n_tokens), dim=-1), 
            requires_grad=False,
        )
    
    def forward(self, x):
        """x: [b h w d] -> [b h w d]"""
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        x = torch.einsum('mn, bnc -> bmc', self.random_matrix, x)
        x = x.reshape(B, H, W, C)
        return x


class AverageMixing(nn.Module):
    """
    Implementation of pooling for PoolFormer: https://arxiv.org/abs/2111.11418
    Modified for [B, H, W, C] input.
    """
    def __init__(self, size, emb_dim, kernel_size):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2, count_include_pad=False)

    def forward(self, x):
        """x: [b h w d] -> [b h w d]"""
        y = x.permute(0, 3, 1, 2)
        y = self.pool(y)
        y = y.permute(0, 2, 3, 1)
        return y - x


class SeparableConvolution(nn.Module):
    """
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """
    def __init__(self, size, emb_dim, expansion_ratio=2, kernel_size=7, padding=3, bias=False):
        super().__init__()
        hidden_dim = int(expansion_ratio * emb_dim)
        self.pwconv1 = nn.Linear(emb_dim, hidden_dim, bias=bias)
        self.act1 = StarReLU()
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding, groups=hidden_dim, bias=bias)
        self.pwconv2 = nn.Linear(hidden_dim, emb_dim, bias=bias)

    def forward(self, x):
        """x: [b h w d] -> [b h w d]"""
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv2(x)
        return x


class Attention(nn.Module):

    def __init__(self, size, emb_dim, head_dim=32, dropout=0, assignment=None):
        super().__init__()
        self.assign = {
            None: None,
            'soft': torch.nn.functional.softmax,
            'hard': hardmax,
        }[assignment]

        self.n_heads = emb_dim // head_dim
        self.dropout = dropout
        self.scale = (emb_dim // self.n_heads) ** -0.5
        self.to_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.to_kv = nn.Linear(emb_dim, 2 * emb_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(emb_dim, emb_dim, bias=False),
            nn.Dropout(dropout),
        )
    
    def forward(self, x, clusters=None, token_sizes=None):
        """
        x: [b n d] -> [b n d]
        clusters: [b nc d] -> [b nc d]
        token_sizes: [b n] ->
        """
        if clusters is not None:
            q, ps = pack([x, clusters], 'b * d')
        else:
            q = x
        q = self.to_q(q)
        kv = self.to_kv(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.n_heads)
        k, v = rearrange(kv, 'b n (k h d) -> k b h n d', k=2, h=self.n_heads)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale # [b h n n]

        if clusters is not None:
            [attn, attn_c] = unpack(attn, ps, 'b h * n') # [b h n n], [b h nc n]

            if self.assign is None:
                attn_c = torch.nn.functional.softmax(attn_c, dim=-1)
            else:
                attn_c = self.assign(attn_c, dim=-2)
                attn_c = attn_c / (torch.sum(attn_c, dim=-1, keepdim=True) + 1e-5)
            clusters = torch.matmul(attn_c, v)
            clusters = rearrange(clusters, 'b h nc d -> b nc (h d)')
            clusters = self.to_out(clusters)

        if token_sizes is not None:
            attn = attn + torch.log(token_sizes + 1e-10)[:, None, :, None]
        attn = torch.nn.functional.softmax(attn, dim=-1)
        x = torch.matmul(attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.to_out(x)

        return x, clusters


class AttentionAggregate(nn.Module):

    def __init__(self, emb_dim, head_dim=32, assignment=None):
        super().__init__()
        self.assign = {
            None: None,
            'soft': torch.nn.functional.softmax,
            'hard': hardmax,
        }[assignment]

        self.n_heads = emb_dim // head_dim
        self.scale = (emb_dim // self.n_heads) ** -0.5
        self.to_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.to_kv = nn.Linear(emb_dim, 2 * emb_dim, bias=False)
        self.to_out = nn.Linear(emb_dim, emb_dim, bias=False)
    
    def forward(self, x, clusters):
        """
        x: [b n d] -> [b n d]
        clusters: [b nc d] -> [b nc d]
        token_sizes: -> [b nc]
        """
        q = self.to_q(clusters)
        kv = self.to_kv(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.n_heads)
        k, v = rearrange(kv, 'b n (k h d) -> k b h n d', k=2, h=self.n_heads)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale # [b h nc n]
        if self.assign is None:
            attn = torch.nn.functional.softmax(attn, dim=-1)
            token_sizes = None
        else:
            attn = self.assign(attn, dim=-2)
            token_sizes = attn.sum(-1) # [b h nc n] -> [b h nc]
            attn = attn / (token_sizes[..., None] + 1e-6)
            token_sizes = token_sizes.mean(1) # [b nc]
        clusters = torch.matmul(attn, v)
        clusters = rearrange(clusters, 'b h n d -> b n (h d)')
        clusters = self.to_out(clusters)

        return clusters, token_sizes


def create_token_mixer(name, pooling, size, emb_dim, *args, **kwargs):
    model = {
        'identity': IdentityMixing,
        'random': RandomMixing,
        'avg': AverageMixing,
        'conv': SeparableConvolution,
        'attention': Attention,
    }[name]
    if pooling is not None and pooling.need_clusters:
        kwargs['assignment'] = pooling.assignment
    return model(size, emb_dim, *args, **kwargs)


class MLP(nn.Module):
    
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0., bias=False):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = drop if isinstance(drop, (list, tuple)) else (drop, drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MetaFormerBlock(nn.Module):
    
    def __init__(
        self,
        size,
        emb_dim,
        token_mixer,
        pooling,
        block_idx,
        norm_layer,
        drop=0.,
        drop_path=0.,
        res_scale_init_value=None,
    ):
        super().__init__()
        
        self.use_attn = token_mixer['name'] == 'attention'
        self.compute_clusters = pooling is not None and pooling.need_clusters
        if self.compute_clusters:
            assert self.use_attn
            if block_idx == 0:
                if pooling.clusters_init is None:
                    n_tokens = utils.product(size)
                    n_clusters = 2 * (n_tokens // 4)
                    self.clusters = nn.Parameter(torch.zeros(n_clusters, emb_dim))
                else:
                    half_size = [s // 2 for s in size]
                    self.init_clusters = create_pooling(
                        pooling.clusters_init['name'],
                        emb_dim,
                        2 * emb_dim, size,
                        half_size,
                        **utils.filter_dict(pooling.clusters_init, ['name']),
                    )

        self.norm1 = norm_layer(emb_dim)
        self.token_mixer = create_token_mixer(token_mixer['name'], pooling, size, emb_dim, **utils.filter_dict(token_mixer, ['name']))
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.res_scale1 = Scale(dim=emb_dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(emb_dim)
        self.mlp = MLP(dim=emb_dim, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.res_scale2 = Scale(dim=emb_dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        if hasattr(self, 'clusters'):
            nn.init.normal_(self.clusters, mean=0, std=0.02)
        
    def forward(self, x, clusters, token_sizes):
        """
        x: [b h w d] -> [b h w d]
        clusters: [b nc d] -> [b nc d]
        token_sizes: [b h w] ->
        """
        # Mix tokens
        if self.use_attn:        
            # Init clusters
            if self.compute_clusters and clusters is None:
                if hasattr(self, 'clusters'):
                    clusters = repeat(self.clusters, 'n d -> b n d', b=x.shape[0])
                else:
                    clusters = self.init_clusters(x) # [b h w d] -> [b h/2 w/2 (2 d)]
                    clusters = rearrange(clusters, 'b h w (two d) -> b (h w two) d', two=2)

            # Pack x for attention: [b h w d] -> [b n d]
            x, ps = pack([x], 'b * d')
            if token_sizes is not None:
                token_sizes, _ = pack([token_sizes], 'b *')
            normed_clusters = None if clusters is None else self.norm1(clusters)
            mixed, mixed_c = self.token_mixer(self.norm1(x), normed_clusters, token_sizes)
            if self.compute_clusters:
                x, ps_cat = pack([x, clusters], 'b * d')
                mixed, _ = pack([mixed, mixed_c], 'b * d')
        else:
            mixed = self.token_mixer(self.norm1(x))

        x = self.res_scale1(x) + self.drop_path1(mixed)
        x = self.res_scale2(x) + self.drop_path2(self.mlp(self.norm2(x)))

        if self.use_attn:
            if self.compute_clusters:
                [x, clusters] = unpack(x, ps_cat, 'b * d')
            [x] = unpack(x, ps, 'b * d')
            if token_sizes is not None:
                [token_sizes] = unpack(token_sizes, ps, 'b *')

        return x, clusters


def create_norm(norm_spatial):
    if norm_spatial:
        return partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False)
    return partial(LayerNormWithoutBias, eps=1e-6)


class MetaFormer(nn.Module):
    
    def __init__(
        self,
        in_channels,
        in_size,
        patch_size,
        emb_dims,
        depths,
        stem,
        poolings,
        token_mixers,
        norm_spatial,
        drop_path,
        res_scale_init_values=[None, None, 1.0, 1.0],
    ):
        super().__init__()

        if not isinstance(depths, (list, tuple)):
            depths = [depths] # it means the model has only one stage
        if not isinstance(emb_dims, (list, tuple)):
            emb_dims = [emb_dims]

        n_stages = len(depths)
        self.n_stages = n_stages

        size0 = utils.ceil_div_it(in_size, patch_size)
        sizes = [[s // (2 ** i) for s in size0] for i in range(n_stages)]

        self.stem = create_stem(
            stem['name'], in_channels, in_size, patch_size, emb_dims[0], add_sin_posemb=False, **utils.filter_dict(stem, ['name']),
        )

        self.poolings = nn.ModuleList([create_pooling(
            pooling['name'], emb_dims[i], emb_dims[i + 1], sizes[i], sizes[i + 1], **utils.filter_dict(pooling, ['name']),
        ) for i, pooling in enumerate(poolings)])
        self.poolings.append(None)
        
        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * n_stages

        norm_layer = create_norm(norm_spatial)
        
        dp_rates = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]

        if not isinstance(res_scale_init_values, (list, tuple)):
            res_scale_init_values = [res_scale_init_values] * n_stages

        self.stages = nn.ModuleList() # each stage consists of multiple metaformer blocks
        curr = 0
        for i in range(n_stages):
            stage = nn.ModuleList([
                MetaFormerBlock(
                    size=sizes[i],
                    emb_dim=emb_dims[i],
                    token_mixer=token_mixers[i],
                    pooling=self.poolings[i],
                    block_idx=j,
                    norm_layer=norm_layer,
                    drop_path=dp_rates[curr + j],
                    res_scale_init_value=res_scale_init_values[i],
                ) for j in range(depths[i])
            ])
            self.stages.append(stage)
            curr += depths[i]
        
        self.out_dim = emb_dims[-1]
        self.head_pre_norm = lambda: nn.LayerNorm(self.out_dim, eps=1e-6)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """x: [b c H W] -> [b d]"""
        x, _ = self.stem(x) # [b c H W] -> [b h w d]

        token_sizes = None
        for i in range(self.n_stages):
            clusters = None
            for block in self.stages[i]:
                x, clusters = block(x, clusters, token_sizes)
            if i != self.n_stages - 1:
                args = [] if clusters is None else [clusters]
                x = self.poolings[i](x, *args)
                if self.poolings[i].output_token_sizes: x, token_sizes = x

        # Global avg pool
        if token_sizes is not None:
            token_sizes = token_sizes / token_sizes.sum(1, keepdim=True).sum(2, keepdim=True) # [b h w]
            x = x * token_sizes.unsqueeze(-1) # [b h w d]
            reductor = torch.sum
        else:
            reductor = torch.mean
        x = torch.movedim(x, -1, 1) # [b h w d] -> [h d h w]
        x = reductor(torch.flatten(x, start_dim=2), dim=-1) # [b d h w] -> [b d]

        return x


def get_args_parser():
    parser = ArgumentParser("metaformer", add_help=False)
    parser.add_argument("--patch_size") # list[int]

    parser.add_argument("--emb_dims") # list[int]
    parser.add_argument("--depths") # list[int]

    parser.add_argument("--stem", type=dict)
    parser.add_argument("--poolings") # list[dict]
    parser.add_argument("--token_mixers") # list[dict]

    parser.add_argument("--norm_spatial", type=bool)
    parser.add_argument("--drop_path", type=float)

    return parser


def create_encoder(config_root, config, in_channels, in_size, n_classes):
    args = get_args(config_root, (get_args_parser(), config))
    model = MetaFormer(
        in_channels,
        in_size,
        args.patch_size,
        args.emb_dims,
        args.depths,
        args.stem,
        args.poolings,
        args.token_mixers,
        args.norm_spatial,
        args.drop_path,
    )
    model.args = args
    return model