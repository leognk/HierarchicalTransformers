# My implementation of Swin.

import torch.nn as nn
from einops import pack, unpack, rearrange, repeat
from .transformer import Transformer
from ..stems.linear import Linear as LinearStem
import utils
from utils import ArgumentParser, get_args
from ..utils import *


class GroupPadder(nn.Module):
    """Pad input tensor so that it can be split into groups of sizes group_size."""

    def __init__(self, group_size):
        super().__init__()
        self.group_size = group_size

    def forward(self, x, real_size):
        pd = get_real_missing_pad(x.shape[1:-1], real_size, self.group_size)
        pd = to_torch_pad(pd, emb_dim_last=True)
        x = do_pad(x, pd)
        return x, pd


class GroupSplitter(nn.Module):

    def __init__(self, group_size):
        super().__init__()
        self.padder = GroupPadder(group_size)
        self.splitter = RearrangeNd('b [(g0 n0)] d -> b [g0] ([n0]) d', {'n': group_size})
    
    def forward(self, x, real_size):
        x, pd = self.padder(x, real_size)
        x = self.splitter(x)
        x, ps = pack([x], '* n d')
        return x, ps, pd


class GroupMerger(nn.Module):

    def __init__(self, group_size):
        super().__init__()
        self.merger = RearrangeNd('b [g0] ([n0]) d -> b [(g0 n0)] d', {'n': group_size})
    
    def forward(self, x, ps, pd):
        [x] = unpack(x, ps, '* n d')
        x = self.merger(x)
        x = do_pad(x, inv_pad(pd))
        return x


class ShiftWindow(nn.Module):

    def __init__(self, in_size, ctx_size):
        super().__init__()

        self.ctx_size = ctx_size
        n_axes = len(ctx_size)
        self.dims = tuple(range(1, n_axes + 1))
        self.shifts = tuple(-(s // 2) for s in ctx_size)
        self.unshifts = tuple(-s for s in self.shifts)
        self.pd = to_torch_pad([True, ctx_size], emb_dim_last=True)

        self.merge = RearrangeNd('b [g0] ([n0]) d -> b [(g0 n0)] d', {'n': ctx_size})
        self.split = RearrangeNd('b [(g0 n0)] d -> b [g0] ([n0]) d', {'n': ctx_size})

        # Attention mask
        attn_mask = torch.zeros(in_size) # [N1 N2]
        slices0 = (slice(0, -ctx_size[0]), slice(-ctx_size[0], self.shifts[0]), slice(self.shifts[0], None))
        slices1 = (slice(0, -ctx_size[1]), slice(-ctx_size[1], self.shifts[1]), slice(self.shifts[1], None))
        idx = 0
        for s0 in slices0:
            for s1 in slices1:
                attn_mask[s0, s1] = idx
                idx += 1
        attn_mask = rearrange(
            attn_mask, '(g1 n1) (g2 n2) -> (g1 g2) (n1 n2)', n1=ctx_size[0], n2=ctx_size[1],
        ) # [N1 N2] -> [g n]
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2) # [g n] -> [g n n]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100) # [g n n] -> [g n n]
        self.register_buffer("attn_mask", attn_mask) # [g n n]
    
    def shift(self, x, ps):
        """x: [B n d] -> [B n d]"""
        [x] = unpack(x, ps, '* n d') # [B n d] -> [b g1 ... gd n d]
        x = self.merge(x) # [b g1 ... gd n d] -> [b N1 ... Nd d]

        x = torch.roll(x, shifts=self.shifts, dims=self.dims)

        x = self.split(x) # [b N1 ... Nd d] -> [b g1 ... gd n d]
        x, self.ps = pack([x], '* n d') # [b g1 ... gd n d] -> [B n d]
        return x
    
    def unshift(self, x):
        """x: [B n d] -> [B n d]"""
        [x] = unpack(x, self.ps, '* n d') # [B n d] -> [b g1 ... gd n d]
        x = self.merge(x) # [b g1 ... gd n d] -> [b N1 ... Nd d]

        x = torch.roll(x, shifts=self.unshifts, dims=self.dims)

        x = self.split(x) # [b N1 ... Nd d] -> [b g1 ... gd n d]
        x, _ = pack([x], '* n d') # [b g1 ... gd n d] -> [B n d]
        return x


class Downsample(nn.Module):

    def __init__(self, emb_dim1, emb_dim2):
        super().__init__()
        self.norm = nn.LayerNorm(4 * emb_dim1)
        self.proj = nn.Linear(4 * emb_dim1, emb_dim2, bias=False)
    
    def forward(self, x):
        """[b (n1 2) (n2 2) d1] -> [b n1 n2 d2]"""
        x = rearrange(x, 'b (n1 k1) (n2 k2) d1 -> b n1 n2 (k1 k2 d1)', k1=2, k2=2)
        x = self.proj(self.norm(x)) # [b n1 n2 (4 d1)] -> [b n1 n2 d2]
        return x


class Stage(nn.Module):

    def __init__(
        self,
        in_size,
        ctx_size,
        emb_dim,
        mlp_hidden_ratio,
        n_heads,
        n_layers,
        dropout,
        drop_path,
        use_flash_attn,
        add_rel_pos,
        downsample,
    ):
        super().__init__()

        self.splitter = GroupSplitter(ctx_size)
        self.merger = GroupMerger(ctx_size)
        self.shift_window = ShiftWindow(in_size, ctx_size)
        rel_pos_ctx_size = ctx_size if add_rel_pos else None
        self.transformers = nn.ModuleList([
            Transformer(
                1, True, emb_dim, int(mlp_hidden_ratio * emb_dim), n_heads, dropout, [drop_path[i]], use_flash_attn, rel_pos_ctx_size,
            ) for i in range(n_layers)
        ])
        self.downsample = Downsample(emb_dim, 2 * emb_dim) if downsample else None
    
    def forward(self, x, real_size):
        """x: [b N1 ... Nd d] -> [b N1' ... Nd' d']"""
        b = x.shape[0]
        x, ps, pd = self.splitter(x, real_size) # [b N1 ... Nd d] -> [B n d]

        for i, layer in enumerate(self.transformers):
            shift = i % 2 == 1
            attn_mask = None
            if shift:
                x = self.shift_window.shift(x, ps)
                attn_mask = self.shift_window.attn_mask # [g n n]
                attn_mask = repeat(attn_mask, 'g n1 n2 -> (b g) n1 n2', b=b) # [g n n] -> [B n n]
            x = layer(x, attn_mask=attn_mask) # [B n d] -> [B n d]
            if shift:
                x = self.shift_window.unshift(x)

        x = self.merger(x, ps, pd) # [B n d] -> [b N1 ... Nd d]

        if self.downsample:
            x = self.downsample(x) # [b N1 ... Nd d] -> [b N1' ... Nd' d']
            real_size = utils.mult_it((0.5,) * len(real_size), real_size)
        return x, real_size


class Backbone(nn.Module):

    def __init__(
        self,
        in_size,
        ctx_sizes,
        emb_dim,
        mlp_hidden_ratios,
        n_heads,
        n_layers_stages,
        dropout,
        drop_path,
        use_flash_attn,
        add_rel_pos,
    ):
        super().__init__()
        self.n_layers = len(ctx_sizes)
        in_sizes = [[s // (2 ** i) for s in in_size] for i in range(self.n_layers)]
        emb_dims = [emb_dim * (2 ** i) for i in range(self.n_layers)]
        self.out_dim = emb_dims[-1]
        drop_path_stages = self.get_drop_path_stages(drop_path, n_layers_stages)
        self.stages = nn.ModuleList([Stage(
            in_sizes[i],
            ctx_sizes[i],
            emb_dims[i],
            mlp_hidden_ratios[i],
            n_heads[i],
            n_layers_stages[i],
            dropout,
            drop_path_stages[i],
            use_flash_attn,
            add_rel_pos,
            i < self.n_layers - 1,
        ) for i in range(self.n_layers)])
        self._initialize_weights()
    
    def _initialize_weights(self):
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    @staticmethod
    def get_drop_path_stages(drop_path, n_layers_stages):
        """output: stage x layer"""
        n_layers = sum(n_layers_stages)
        dpr = torch.linspace(0, drop_path, n_layers).tolist()
        res = []
        i = 0
        for s in n_layers_stages:
            i1 = i + s
            res.append(dpr[i:i1])
            i = i1
        return res
    
    def forward(self, x):
        real_size = tuple(x.shape[1:-1])
        for stage in self.stages:
            x, real_size = stage(x, real_size)
        return x
    
    @property
    def num_lr_scale_groups(self):
        return self.n_layers
    
    def lr_scale_group_id(self, name):
        layer_id = int(name.split('.')[1])
        return layer_id


class Swin(nn.Module):

    def __init__(
        self,
        in_channels,
        in_size,
        patch_size,
        ctx_sizes,
        emb_dim,
        mlp_hidden_ratios,
        n_heads,
        n_layers_stages,
        dropout,
        drop_path,
        use_flash_attn,
        add_abs_pos,
        add_rel_pos,
    ):
        super().__init__()
        
        self.stem = LinearStem(in_channels, in_size, patch_size, emb_dim, add_sin_posemb=add_abs_pos)
        self.backbone = Backbone(
            utils.ceil_div_it(in_size, patch_size),
            ctx_sizes,
            emb_dim,
            mlp_hidden_ratios,
            n_heads,
            n_layers_stages,
            dropout,
            drop_path,
            use_flash_attn,
            add_rel_pos,
        )
        self.out_dim = self.backbone.out_dim
        self.head_pre_norm = lambda: nn.LayerNorm(self.out_dim)
    
    def forward(self, x):
        x, _ = self.stem(x)
        x = self.backbone(x)
        return x
    
    @property
    def no_weight_decay(self):
        nwd = {"rel_pos_bias_table"}
        res = set()
        for n, _ in self.named_parameters():
            if any(x in n for x in nwd):
                res.add(n)
        return res
    
    @property
    def num_lr_scale_groups(self):
        return 1 + self.backbone.num_lr_scale_groups
    
    def lr_scale_group_id(self, name):
        if name.startswith("stem"):
            return 0
        elif name.startswith("backbone"):
            _name = name.split('.', 1)[1]
            return 1 + self.backbone.lr_scale_group_id(_name)
        else:
            raise ValueError(f"Invalid parameter name: {name}")


def get_args_parser():
    parser = ArgumentParser("swin", add_help=False)
    parser.add_argument("--patch_size") # list[int]

    parser.add_argument("--ctx_sizes") # list[list[int]]

    parser.add_argument("--emb_dim") # int
    parser.add_argument("--mlp_hidden_ratios") # list[int]
    parser.add_argument("--n_heads") # list[int]

    parser.add_argument("--n_layers_stages") # list[int]

    parser.add_argument("--dropout", type=float)
    parser.add_argument("--drop_path", type=float)
    parser.add_argument("--use_flash_attn", type=bool)

    parser.add_argument("--add_abs_pos", type=bool)
    parser.add_argument("--add_rel_pos", type=bool)

    return parser


def create_encoder(config_root, config, in_channels, in_size, n_classes):
    args = get_args(config_root, (get_args_parser(), config))
    model = Swin(
        in_channels,
        in_size,
        args.patch_size,
        args.ctx_sizes,
        args.emb_dim,
        args.mlp_hidden_ratios,
        args.n_heads,
        args.n_layers_stages,
        args.dropout,
        args.drop_path,
        args.use_flash_attn,
        args.add_abs_pos,
        args.add_rel_pos,
    )
    model.args = args
    return model