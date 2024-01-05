import torch.nn as nn
from einops import repeat, pack, unpack, rearrange
from functools import partial
from .transformer import MLP, Transformer, Attention
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
        return x


class GroupSplitter(nn.Module):

    def __init__(self, group_size):
        super().__init__()
        self.padder = GroupPadder(group_size)
        self.splitter = RearrangeNd('b [(g0 n0)] d -> b [g0] ([n0]) d', {'n': group_size})
    
    def forward(self, x, real_size):
        x = self.padder(x, real_size)
        x = self.splitter(x)
        x, ps = pack([x], '* n d')
        return x, ps


class GroupMerger(nn.Module):

    def __init__(self, group_size):
        super().__init__()
        self.merger = RearrangeNd('b [g0] ([n0]) d -> b [(g0 n0)] d', {'n': group_size})
    
    def forward(self, x, ps):
        [x] = unpack(x, ps, '* n d')
        x = self.merger(x)
        return x


class LearnableQueryInitializer(nn.Module):
    """Initialize query with learnable embeddings."""

    def __init__(self, ctx_size, qry_size, in_dim, out_dim):
        super().__init__()
        n_tokens = utils.product(qry_size)
        self.query = nn.Parameter(torch.zeros(n_tokens, out_dim))
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.normal_(self.query, mean=0, std=0.02)
    
    def forward(self, context):
        """[b n d] -> [b n' d']"""
        return repeat(self.query, 'n d -> b n d', b=context.shape[0])


class PosEmbQueryInitializer(nn.Module):
    """Initialize query with sinusoidal positional embeddings passed through an MLP."""

    def __init__(self, ctx_size, qry_size, in_dim, out_dim, temperature=10000):
        super().__init__()
        self.register_buffer(
            "posemb",
            get_nd_sin_posemb(qry_size, out_dim, temperature=temperature),
        )
        self.mlp = MLP(out_dim, out_dim, 0)
    
    def forward(self, context):
        """[b n d] -> [b n' d']"""
        posemb, _ = pack([self.posemb], '* d')
        query = self.mlp(posemb)
        return repeat(query, 'n d -> b n d', b=context.shape[0])


class CtxProjQueryInitializer(nn.Module):
    """Initialize query with a learnable n-dimensional weighted sum of the context tokens."""

    def __init__(self, ctx_size, qry_size, in_dim, out_dim):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim

        self.split = RearrangeNd('b ([n0]) d -> b [n0] d', {'n': ctx_size})
        self.merge = RearrangeNd('b [n0] d -> b ([n0]) d', {'n': qry_size})
        proj_i = lambda i: nn.Sequential(
            Transpose(-i - 2, -1),
            nn.Linear(ctx_size[-i - 1], qry_size[-i - 1]),
            Transpose(-i - 2, -1),
        )
        self.projs = nn.Sequential(*[proj_i(i) for i in range(len(ctx_size))])
        if in_dim != out_dim:
            self.scale_dim = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, out_dim),
            )
    
    def forward(self, context):
        """[b n d] -> [b n' d']"""
        x = self.split(context) # [b n d] -> [b n1 ... nd d]
        x = self.projs(x) # [b n1 ... nd d] -> [b n1' ... nd' d]
        if self.in_dim != self.out_dim:
            x = self.scale_dim(x) # [b n1' ... nd' d] -> [b n1' ... nd' d']
        x = self.merge(x) # [b n1' ... nd' d'] -> [b n' d']
        return x


class LinearPoolQueryInitializer(nn.Module):

    def __init__(self, ctx_size, qry_size, in_dim, out_dim):
        super().__init__()
        assert len(ctx_size) == len(qry_size) == 2
        self.split = RearrangeNd('b ([n0]) d -> b [n0] d', {'n': ctx_size})
        self.merge = RearrangeNd('b [n0] d -> b ([n0]) d', {'n': qry_size})
        self.norm = nn.LayerNorm(4 * in_dim)
        self.proj = nn.Linear(4 * in_dim, out_dim, bias=False)
    
    def forward(self, context):
        """[b n d] -> [b n' d']"""
        x = self.split(context) # [b n d] -> [b n1 n2 d]
        x = rearrange(x, 'b (n1 k1) (n2 k2) d -> b n1 n2 (k1 k2 d)', k1=2, k2=2)
        x = self.proj(self.norm(x)) # [b n1' n2' (4 d)] -> [b n1' n2' d']
        x = self.merge(x) # [b n1' n2' d'] -> [b n' d']
        return x


class PoolQueryInitializer(nn.Module):

    def __init__(self, ctx_size, qry_size, in_dim, out_dim, method):
        super().__init__()
        assert all(c % q == 0 for c, q in zip(ctx_size, qry_size))

        self.split = RearrangeNd('b ([n0]) d -> b [n0] d', {'n': ctx_size})
        self.merge = RearrangeNd('b [n0] d -> b ([n0]) d', {'n': qry_size})
        
        self.norm = nn.LayerNorm((in_dim, *ctx_size))

        stride = tuple(c // q for c, q in zip(ctx_size, qry_size))
        kernel_size = tuple(2 * s - 1 for s in stride)
        padding = tuple((k - 1) // 2 for k in kernel_size)

        if method == 'conv':
            self.pool = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding)
        elif method == 'avg':
            self.pool = nn.Sequential(
                nn.AvgPool2d(kernel_size, stride, padding),
                nn.Conv2d(in_dim, out_dim, 1, 1, 0),
            )
        elif method == 'max':
            self.pool = nn.Sequential(
                nn.MaxPool2d(kernel_size, stride, padding),
                nn.Conv2d(in_dim, out_dim, 1, 1, 0),
            )
        else:
            raise ValueError(f"Invalid method: {method}")
    
    def forward(self, context):
        """[b n d] -> [b n' d']"""
        x = self.split(context) # [b n d] -> [b n1 n2 d]
        x = torch.movedim(x, -1, 1) # [b n1 n2 d] -> [b d n1 n2]
        x = self.pool(self.norm(x)) # [b d n1 n2] -> [b d' n1' n2']
        x = torch.movedim(x, 1, -1) # [b d' n1' n2'] -> [b n1' n2' d']
        x = self.merge(x) # [b n1' n2' d'] -> [b n' d']
        return x


class CtxSubsetQueryInitializer(nn.Module):

    def __init__(self, ctx_size, qry_size, in_dim, out_dim):
        super().__init__()
        assert len(ctx_size) == len(qry_size) == 2
        self.split = RearrangeNd('b ([n0]) d -> b [n0] d', {'n': ctx_size})
        self.merge = RearrangeNd('b [n0] d -> b ([n0]) d', {'n': qry_size})
        self.in_dim, self.out_dim = in_dim, out_dim
        if in_dim != out_dim:
            self.scale_dim = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, out_dim),
            )
    
    def forward(self, context):
        """[b n d] -> [b n' d']"""
        x = self.split(context) # [b n d] -> [b n1 n2 d]
        x = x[:, ::2, ::2, :] # [b n1 n2 d] -> [b n1' n2' d]
        x = self.merge(x) # [b n1' n2' d] -> [b n' d]
        if self.in_dim != self.out_dim:
            x = self.scale_dim(x) # [b n' d] -> [b n' d']
        return x


class SelfAttentionPrune(nn.Module):

    def __init__(self, emb_dim, n_heads, n_keep):
        super().__init__()
        assert emb_dim % n_heads == 0
        self.n_heads = n_heads
        self.n_keep = n_keep
        self.scale = (emb_dim // n_heads) ** -0.5
        self.to_qkv = nn.Linear(emb_dim, 3 * emb_dim)
        self.to_out = nn.Linear(emb_dim, emb_dim)
    
    def forward(self, x):
        x0 = x.clone()

        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (k h d) -> k b h n d', k=3, h=self.n_heads)
        a = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        a = torch.nn.functional.softmax(a, dim=-1) # [b h n n]

        s = torch.sum(a, dim=-2) # [b h n]
        idx = torch.topk(s, self.n_keep, dim=-1)[1] # [b h n']
        a = torch.gather(a, dim=-2, index=repeat(idx, 'b h nq -> b h nq nk', nk=a.shape[-1])) # [b h n' n]

        x = torch.matmul(a, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.to_out(x)

        x0 = rearrange(x0, 'b n (h d) -> b h n d', h=self.n_heads)
        x0 = torch.gather(x0, dim=-2, index=repeat(idx, 'b h n -> b h n d', d=x0.shape[-1])) # [b h n' d]
        x0 = rearrange(x0, 'b h n d -> b n (h d)')
        x += x0
        return x


class SAPruneQueryInitializer(nn.Module):

    def __init__(self, ctx_size, qry_size, in_dim, out_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = SelfAttentionPrune(in_dim, in_dim // 32, utils.product(qry_size))
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = MLP(in_dim, 4 * in_dim, 0.0)
        self.in_dim, self.out_dim = in_dim, out_dim
        if in_dim != out_dim:
            self.scale_dim = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, out_dim),
            )
    
    def forward(self, x):
        """[b n d] -> [b n' d']"""
        x = self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        if self.in_dim != self.out_dim:
            x = self.scale_dim(x) 
        return x


class QueryInitializer(nn.Module):

    def __init__(self, method, ctx_size, qry_size, in_dim, out_dim):
        super().__init__()
        self.initializer = {
            'learnable': LearnableQueryInitializer,
            'posemb': PosEmbQueryInitializer,
            'ctx_proj': CtxProjQueryInitializer,
            'linear_pool': LinearPoolQueryInitializer,
            'avg_pool': partial(PoolQueryInitializer, method='avg'),
            'max_pool': partial(PoolQueryInitializer, method='max'),
            'conv': partial(PoolQueryInitializer, method='conv'),
            'ctx_subset': CtxSubsetQueryInitializer,
            'sa_prune': SAPruneQueryInitializer,
        }[method](ctx_size, qry_size, in_dim, out_dim)
    
    def forward(self, context):
        """[b n d] -> [b n' d']"""
        x = self.initializer(context)
        return x


class GroupShift(nn.Module):

    def __init__(self, ctx_size, qry_size):
        super().__init__()

        n_axes = len(ctx_size)
        self.dims = tuple(range(1, n_axes + 1))

        self.shifts_ctx = tuple(s // 2 for s in ctx_size)
        self.shifts_qry = tuple(-(s // 2) for s in qry_size)

        self.merge_ctx = RearrangeNd('b [g0] ([n0]) d -> b [(g0 n0)] d', {'n': ctx_size})
        self.merge_qry = RearrangeNd('b [g0] ([n0]) d -> b [(g0 n0)] d', {'n': qry_size})
        self.split_ctx = RearrangeNd('b [(g0 n0)] d -> b [g0] ([n0]) d', {'n': ctx_size})
        self.split_qry = RearrangeNd('b [(g0 n0)] d -> b [g0] ([n0]) d', {'n': qry_size})
    
    def shift_ctx(self, x, ps):
        """x: [B n d] -> [B n d]"""
        [x] = unpack(x, ps, '* n d') # [B n d] -> [b g1 ... gd n d]
        x = self.merge_ctx(x) # [b g1 ... gd n d] -> [b N1 ... Nd d]
        x = torch.roll(x, shifts=self.shifts_ctx, dims=self.dims)
        x = self.split_ctx(x) # [b N1 ... Nd d] -> [b g1 ... gd n d]
        x, _ = pack([x], '* n d') # [b g1 ... gd n d] -> [B n d]
        return x
    
    def unshift_qry(self, x, ps):
        """x: [B n d] -> [B n d]"""
        [x] = unpack(x, ps, '* n d') # [B n d] -> [b g1 ... gd n d]
        x = self.merge_qry(x) # [b g1 ... gd n d] -> [b N1 ... Nd d]
        x = torch.roll(x, shifts=self.shifts_qry, dims=self.dims)
        x = self.split_qry(x) # [b N1 ... Nd d] -> [b g1 ... gd n d]
        x, _ = pack([x], '* n d') # [b g1 ... gd n d] -> [B n d]
        return x


class Stage(nn.Module):

    def __init__(
        self,
        ctx_size,
        qry_size,
        in_dim,
        out_dim,
        mlp_hidden_ratio,
        in_heads,
        out_heads,
        n_layers,
        qry_init_method,
        dropout,
        drop_path,
        use_flash_attn,
        add_rel_pos,
        shift_groups,
    ):
        super().__init__()

        self.use_qry = qry_size is not None
        self.do_scale_dim = in_dim != out_dim
        self.shift_groups = shift_groups
        if self.use_qry:
            assert len(ctx_size) == len(qry_size)
            assert len(n_layers) == 2
            use_cross_attn = n_layers[1] > 0
            self.do_scale_dim = self.do_scale_dim and use_cross_attn
            self.shift_groups = shift_groups and use_cross_attn
        else:
            assert len(n_layers) == 1
            qry_size = ctx_size

        self.ctx_size, self.qry_size = ctx_size, qry_size
        self.scale_factor = utils.div_it(qry_size, ctx_size)

        self.splitter = GroupSplitter(ctx_size)
        self.merger = GroupMerger(qry_size)
        rel_pos_ctx_size = ctx_size if add_rel_pos else None
        self.self_attn = Transformer(
            n_layers[0], True, in_dim, int(mlp_hidden_ratio * in_dim), in_heads, dropout, drop_path[0], use_flash_attn, rel_pos_ctx_size,
        )
        if self.do_scale_dim:
            self.scale_dim = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, out_dim),
            )
        if self.use_qry:
            self.qry_initializer = QueryInitializer(qry_init_method, ctx_size, qry_size, in_dim, out_dim)
            if self.shift_groups:
                self.group_shift = GroupShift(ctx_size, qry_size)
            self.cross_attn = Transformer(
                n_layers[1], False, out_dim, int(mlp_hidden_ratio * out_dim), out_heads, dropout, drop_path[1], use_flash_attn,
            )
            # self.cross_attn = Attention(False, out_dim, out_heads, dropout, use_flash_attn)

    
    def forward(self, x, real_size):
        """x: [b N1 ... Nd d] -> [b N1' ... Nd' d']"""
        x, ps = self.splitter(x, real_size) # [b N1 ... Nd d] -> [B n d]
        x = self.self_attn(x) # [B n d] -> [B n d]
        if self.use_qry:
            q = self.qry_initializer(x) # [B n d] -> [B n' d']
        if self.do_scale_dim:
            x = self.scale_dim(x) # [B n d] -> [B n d']
        if self.use_qry:
            if self.shift_groups:
                x = self.group_shift.shift_ctx(x, ps)
            x = self.cross_attn(q, x) # [B n' d'], [B n d'] -> [B n' d']
            if self.shift_groups:
                x = self.group_shift.unshift_qry(x, ps)
        x = self.merger(x, ps) # [B n' d'] -> [b N1' ... Nd' d']
        real_size = utils.mult_it(self.scale_factor, real_size)
        return x, real_size


class Backbone(nn.Module):

    def __init__(
        self,
        ctx_sizes,
        qry_sizes,
        emb_dims,
        mlp_hidden_ratios,
        n_heads,
        n_layers_stages,
        qry_init_method,
        dropout,
        drop_path,
        drop_cross_attn,
        use_flash_attn,
        add_rel_pos,
        shift_groups,
    ):
        super().__init__()
        self.n_layers = len(ctx_sizes)
        drop_path_stages = self.get_drop_path_stages(drop_path, drop_cross_attn, n_layers_stages)
        self.stages = nn.ModuleList([Stage(
            ctx_sizes[i],
            qry_sizes[i],
            emb_dims[i],
            emb_dims[i + 1],
            mlp_hidden_ratios[i],
            n_heads[i],
            n_heads[i + 1],
            n_layers_stages[i],
            qry_init_method,
            dropout,
            drop_path_stages[i],
            use_flash_attn,
            add_rel_pos,
            shift_groups,
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
    def get_drop_path_stages(drop_path, drop_cross_attn, n_layers_stages):
        """output: stage x (self_attn or cross_attn) x layer"""
        n_layers = sum(s[0] + s[1] if len(s) == 2 and drop_cross_attn else s[0] for s in n_layers_stages)
        dpr = torch.linspace(0, drop_path, n_layers).tolist()
        res = []
        i = 0
        for s in n_layers_stages:
            i1 = i + s[0]
            self_attn = dpr[i:i1]
            i = i1
            if len(s) == 2 and drop_cross_attn:
                i1 = i + s[1]
                cross_attn = dpr[i:i1]
                i = i1
            else:
                cross_attn = None
            res.append([self_attn, cross_attn])
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


class HViT2(nn.Module):

    def __init__(
        self,
        in_channels,
        in_size,
        patch_size,
        ctx_sizes,
        qry_sizes,
        emb_dims,
        mlp_hidden_ratios,
        n_heads,
        n_layers_stages,
        qry_init_method,
        dropout,
        drop_path,
        drop_cross_attn,
        use_flash_attn,
        add_abs_pos,
        add_rel_pos,
        shift_groups,
    ):
        super().__init__()
        
        self.stem = LinearStem(in_channels, in_size, patch_size, emb_dims[0], add_sin_posemb=add_abs_pos)
        self.backbone = Backbone(
            ctx_sizes,
            qry_sizes,
            emb_dims,
            mlp_hidden_ratios,
            n_heads,
            n_layers_stages,
            qry_init_method,
            dropout,
            drop_path,
            drop_cross_attn,
            use_flash_attn,
            add_rel_pos,
            shift_groups,
        )
        self.out_dim = emb_dims[-1]
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
    parser = ArgumentParser("hvit2", add_help=False)
    parser.add_argument("--patch_size") # list[int]

    parser.add_argument("--ctx_sizes") # list[list[int]]
    parser.add_argument("--qry_sizes") # list[list[int]]

    parser.add_argument("--emb_dims") # list[int]
    parser.add_argument("--mlp_hidden_ratios") # list[int]
    parser.add_argument("--n_heads") # list[int]

    parser.add_argument("--n_layers_stages") # list[list[int]]

    parser.add_argument("--qry_init_method", type=str)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--drop_path", type=float)
    parser.add_argument("--drop_cross_attn", type=bool)
    parser.add_argument("--use_flash_attn", type=bool)

    parser.add_argument("--add_abs_pos", type=bool)
    parser.add_argument("--add_rel_pos", type=bool)

    parser.add_argument("--shift_groups", type=bool)

    return parser


def create_encoder(config_root, config, in_channels, in_size, n_classes):
    args = get_args(config_root, (get_args_parser(), config))
    model = HViT2(
        in_channels,
        in_size,
        args.patch_size,
        args.ctx_sizes,
        args.qry_sizes,
        args.emb_dims,
        args.mlp_hidden_ratios,
        args.n_heads,
        args.n_layers_stages,
        args.qry_init_method,
        args.dropout,
        args.drop_path,
        args.drop_cross_attn,
        args.use_flash_attn,
        args.add_abs_pos,
        args.add_rel_pos,
        args.shift_groups,
    )
    model.args = args
    return model