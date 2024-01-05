import torch.nn as nn
from einops import repeat, pack, unpack
from ..transformer import MLP, Transformer
from ...stems.linear import Linear as LinearStem
from .size_dynamics import PipelineDynamics
import utils
from utils import ArgumentParser, get_args
from ...utils import *


class CodecStash:
    """Defines a stash object that can be used to stash information for next stage computation (forward/backward)."""
    
    def __init__(self):
        self.pd = None # pd = (need_pad, pad)
        self.cp = None # cp = (need_crop, crop)


class GroupPadder(nn.Module):
    """Pad input tensor so that it can be split into groups of sizes group_size."""

    def __init__(self, group_size):
        super().__init__()
        self.group_size = group_size
        self.pd = None # pd = (need_pad, pad)

    def forward(self, x, real_size, stash):
        self.pd = stash.pd
        # If don't know whether to pad or not, figure it out.
        if self.pd is None:
            pd = get_real_missing_pad(x.shape[1:-1], real_size, self.group_size)
            self.pd = to_torch_pad(pd, emb_dim_last=True)
        x = do_pad(x, self.pd)
        return x
    
    def setup_stash(self, stash):
        stash.cp = inv_pad(self.pd)


class GroupCropper(nn.Module):
    """Crop input tensor back to its original shape before padding."""

    def __init__(self):
        super().__init__()
        self.cp = None, None
    
    def forward(self, x, stash):
        self.cp = stash.cp
        # By default, don't crop.
        if self.cp is None:
            self.cp = False, None
        x = do_pad(x, self.cp)
        return x
    
    def setup_stash(self, stash):
        stash.pd = inv_pad(self.cp)


class GroupSplitter(nn.Module):

    def __init__(self, group_size):
        super().__init__()
        self.padder = GroupPadder(group_size)
        self.splitter = RearrangeNd('b [(g0 n0)] d -> b [g0] ([n0]) d', {'n': group_size})
    
    def forward(self, x, real_size, stash):
        x = self.padder(x, real_size, stash)
        x = self.splitter(x)
        x, ps = pack([x], '* n d')
        return x, ps
    
    def setup_stash(self, stash):
        self.padder.setup_stash(stash)


class GroupMerger(nn.Module):

    def __init__(self, group_size):
        super().__init__()
        self.cropper = GroupCropper()
        self.merger = RearrangeNd('b [g0] ([n0]) d -> b [(g0 n0)] d', {'n': group_size})
    
    def forward(self, x, stash, ps):
        [x] = unpack(x, ps, '* n d')
        x = self.merger(x)
        x = self.cropper(x, stash)
        return x
    
    def setup_stash(self, stash):
        self.cropper.setup_stash(stash)


class LearnableQueryInitializer(nn.Module):
    """Initialize query with learnable embeddings."""

    def __init__(self, qry_size, emb_dim):
        super().__init__()
        n_tokens = utils.product(qry_size)
        self.query = nn.Parameter(torch.zeros(n_tokens, emb_dim))
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.normal_(self.query, mean=0, std=0.02)
    
    def forward(self, context):
        return repeat(self.query, 'n d -> b n d', b=context.shape[0])


class PosEmbQueryInitializer(nn.Module):
    """Initialize query with sinusoidal positional embeddings passed through an MLP."""

    def __init__(self, qry_size, emb_dim, temperature=10000):
        super().__init__()
        self.emb_dim = emb_dim
        self.register_buffer(
            "posemb",
            get_nd_sin_posemb(qry_size, self.emb_dim, temperature=temperature),
        )
        self.mlp = MLP(emb_dim, emb_dim, 0)
    
    def forward(self, context):
        posemb, _ = pack([self.posemb], '* d')
        query = self.mlp(posemb)
        return repeat(query, 'n d -> b n d', b=context.shape[0])


class CtxProjQueryInitializer(nn.Module):
    """Initialize query with a learnable n-dimensional weighted sum of the context tokens."""

    def __init__(self, ctx_size, qry_size):
        super().__init__()
        self.split = RearrangeNd('b ([n0]) d -> b [n0] d', {'n': ctx_size})
        self.merge = RearrangeNd('b [n0] d -> b ([n0]) d', {'n': qry_size})
        proj_i = lambda i: nn.Sequential(
            Transpose(-i - 2, -1),
            nn.Linear(ctx_size[-i - 1], qry_size[-i - 1]),
            Transpose(-i - 2, -1),
        )
        self.projs = nn.Sequential(*[proj_i(i) for i in range(len(ctx_size))])
    
    def forward(self, context):
        x = self.split(context)
        x = self.projs(x)
        x = self.merge(x)
        return x


class QueryInitializer(nn.Module):

    def __init__(self, method, ctx_size, qry_size, emb_dim):
        super().__init__()
        if method == 'learnable': self.initializer = LearnableQueryInitializer(qry_size, emb_dim)
        elif method == 'posemb': self.initializer = PosEmbQueryInitializer(qry_size, emb_dim)
        elif method == 'ctx_proj': self.initializer = CtxProjQueryInitializer(ctx_size, qry_size)
        else: raise ValueError(f"Unknown query initializing method: {method}")
    
    def forward(self, context):
        x = self.initializer(context) # [b nc d] -> [b nq d]
        return x


class OneStepCodec(nn.Module):
    """
    Codec (encoder-decoder) block performing context transformation
    and context encoding/decoding jointly in a single step.
    """

    def __init__(self, n_layers, attend_to_query, emb_dim, mlp_hidden_ratio, n_heads, dropout, use_flash_attn):
        super().__init__()
        self.attend_to_query = attend_to_query
        self.emb_dim = emb_dim
        mlp_hidden_dim = int(emb_dim * mlp_hidden_ratio)
        self.transformer = Transformer(n_layers, attend_to_query, emb_dim, mlp_hidden_dim, n_heads, dropout, None, use_flash_attn)
    
    def forward(self, x, context):
        """
        When encoding, context is an expansion and x is a reduction query.
        When decoding, context is a reduction and x is an expansion query.
        """
        x, ps = pack([x, context], 'b * d')
        if self.attend_to_query: context = None
        x = self.transformer(x, context)
        x, context = unpack(x, ps, 'b * d')
        return x, context


class TwoStepCodec(nn.Module):
    """
    Codec (encoder-decoder) block performing context transformation
    and context encoding/decoding sequentially in two separate steps.
    When encoding, the context is first transformed, then encoded.
    When decoding, the context is first decoded, then transformed.
    """

    def __init__(self, n_layers_transformer, n_layers_codec, attend_to_query, emb_dim, mlp_hidden_ratio, n_heads, dropout, use_flash_attn):
        super().__init__()
        self.attend_to_query = attend_to_query
        self.emb_dim = emb_dim
        mlp_hidden_dim = int(emb_dim * mlp_hidden_ratio)
        self.transformer = Transformer(n_layers_transformer, True, emb_dim, mlp_hidden_dim, n_heads, dropout, None, use_flash_attn)
        self.codec = Transformer(n_layers_codec, False, emb_dim, mlp_hidden_dim, n_heads, dropout, None, use_flash_attn)
    
    def forward(self, x, context):
        """
        When encoding, context is an expansion and x is a reduction query.
        When decoding, context is a reduction and x is an expansion query.
        """
        encoding = x.shape[-2] <= context.shape[-2] # compare the number of tokens
        if encoding: context = self.transformer(context)
        context2 = pack([x, context], 'b * d')[0] if self.attend_to_query else context
        x = self.codec(x, context2)
        if not encoding: x = self.transformer(x)
        return x, context


class Codec(nn.Module):
    """
    A wrapper around a k-step codec that manages:
    - context splitting-merging into groups,
    - query initialization,
    - expansion/reduction of the embedding dimension,
    - stashing of information for the codec in the next stage.
    
    If n_layers contains:
    - 1 element: use OneStepCodec.
    - 2 elements: use TwoStepCodec.
    """

    def __init__(
        self,
        forward,
        ctx_size,
        qry_size,
        in_dim,
        out_dim,
        mlp_hidden_ratio,
        n_heads,
        n_layers,
        attend_to_query,
        qry_init_method,
        dropout,
        use_flash_attn,
    ):
        super().__init__()
        assert len(ctx_size) == len(qry_size)
        if not forward: # backward
            ctx_size, qry_size = qry_size, ctx_size
            in_dim, out_dim = out_dim, in_dim
        self.ctx_size, self.qry_size = ctx_size, qry_size
        self.encoding = utils.product(qry_size) <= utils.product(ctx_size)
        self.scale_factor = utils.div_it(qry_size, ctx_size)
        self.in_dim, self.out_dim = in_dim, out_dim
        emb_dim = min(in_dim, out_dim)

        self.splitter = GroupSplitter(ctx_size)
        self.merger = GroupMerger(qry_size)
        if forward:
            self.qry_initializer = QueryInitializer(qry_init_method, ctx_size, qry_size, emb_dim)
        KStepCodec = {1: OneStepCodec, 2: TwoStepCodec}[len(n_layers)]
        self.kstep_codec = KStepCodec(*n_layers, attend_to_query, emb_dim, mlp_hidden_ratio, n_heads, dropout, use_flash_attn)
        if in_dim != out_dim:
            self.proj = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, out_dim),
            )
    
    def forward(self, x, context, real_size, stash):
        """
        x shape: [B' n' d]
        context shape: [b N1 ... Nd d]
        """
        context, ps = self.splitter(context, real_size, stash) # [b N1 ... Nd d] -> [B n d]
        if self.in_dim > self.out_dim: context = self.proj(context)
        if x is None: x = self.qry_initializer(context)
        # The query becomes the next context and the context becomes the next query.
        context, x = self.kstep_codec(x, context)
        if self.in_dim < self.out_dim: context = self.proj(context)
        context = self.merger(context, stash, ps) # [B' n' d] -> [b N1' ... Nd' d]
        real_size = utils.mult_it(self.scale_factor, real_size)
        self.setup_stash(stash)
        return context, x, real_size # [b N1' ... Nd' d], [B n d]

    def setup_stash(self, stash):
        self.splitter.setup_stash(stash)
        self.merger.setup_stash(stash)


class NonRecurrentCodecs(nn.Module):
    """Stores the forward and backward codecs with separate parameters for each layer (non-recurrent)."""

    def __init__(self,
        define_backward,
        ctx_sizes,
        qry_sizes,
        emb_dims,
        mlp_hidden_ratios,
        n_heads,
        n_layers_codec,
        attend_to_query,
        codec,
    ):
        super().__init__()
        self.define_backward = define_backward
        self.n_layers = len(ctx_sizes)
        codec_i = lambda forward, i: codec(
            forward,
            ctx_sizes[i],
            qry_sizes[i],
            emb_dims[i],
            emb_dims[i + 1],
            mlp_hidden_ratios[i],
            n_heads[i],
            n_layers_codec[0 if forward else 1][i],
            attend_to_query[0 if forward else 1][i],
        )
        self.forward_codecs = nn.ModuleList([codec_i(True, i) for i in range(self.n_layers)])
        if self.define_backward:
            self.backward_codecs = nn.ModuleList([codec_i(False, i) for i in range(self.n_layers)])
    
    def get(self, step, idx):
        if step == 1:
            return self.forward_codecs[idx]
        elif step == -1:
            assert self.define_backward
            return self.backward_codecs[idx]
        else:
            raise ValueError(f"Invalid step {step}")
    
    @property
    def num_lr_scale_groups(self):
        return self.n_layers
    
    def lr_scale_group_id(self, name):
        layer_id = int(name.split('.')[1])
        return layer_id


class RecurrentCodecs(nn.Module):
    """Stores the forward and backward codecs with shared parameters across layers (recurrent)."""

    def __init__(self, define_backward, codec):
        super().__init__()
        self.define_backward = define_backward
        self.forward_codec = codec(forward=True)
        if self.define_backward:
            self.backward_codec = codec(forward=False)
    
    def get(self, step):
        if step == 1:
            return self.forward_codec
        elif step == -1:
            assert self.define_backward
            return self.backward_codec
        else:
            raise ValueError(f"Invalid step {step}")
    
    @property
    def num_lr_scale_groups(self):
        return 1
    
    def lr_scale_group_id(self, name):
        return 0


class Codecs(nn.Module):
    """
    A wrapper around (Non)RecurrentCodecs storing codecs.
    Handles parameters initialization.
    
    If ctx_sizes contains:
    - 1 element: use RecurrentCodecs.
    - > 1 elements: use NonRecurrentCodecs.
    """

    def __init__(
        self,
        define_backward,
        ctx_sizes,
        qry_sizes,
        emb_dims,
        mlp_hidden_ratios,
        n_heads,
        n_layers_codec,
        attend_to_query,
        qry_init_method,
        dropout,
        use_flash_attn,
    ):
        super().__init__()
        self.n_axis = len(ctx_sizes[0])
        self.recurrent = len(ctx_sizes) == 1
        codec = lambda forward, ctx_size, qry_size, in_dim, out_dim, mlp_hidden_ratio, n_heads, n_layers, attend_to_query:\
            Codec(
                forward,
                ctx_size,
                qry_size,
                in_dim,
                out_dim,
                mlp_hidden_ratio,
                n_heads,
                n_layers,
                attend_to_query,
                qry_init_method,
                dropout,
                use_flash_attn,
            )
        if self.recurrent:
            self.codecs = RecurrentCodecs(define_backward, lambda forward:\
                codec(
                    forward,
                    ctx_sizes[0],
                    qry_sizes[0],
                    emb_dims[0],
                    emb_dims[0],
                    mlp_hidden_ratios[0],
                    n_heads[0],
                    n_layers_codec[0 if forward else 1][0],
                    attend_to_query[0 if forward else 1][0],
                )
            )
            self.n_layers = None
        else:
            self.codecs = NonRecurrentCodecs(
                define_backward,
                ctx_sizes,
                qry_sizes,
                emb_dims,
                mlp_hidden_ratios,
                n_heads,
                n_layers_codec,
                attend_to_query,
                codec,
            )
            self.n_layers = self.codecs.n_layers
    
    def get(self, step, idx):
        if self.recurrent: return self.codecs.get(step)
        else: return self.codecs.get(step, idx)
    
    @property
    def num_lr_scale_groups(self):
        return self.codecs.num_lr_scale_groups
    
    def lr_scale_group_id(self, name):
        _name = name.split('.', 1)[1]
        return self.codecs.lr_scale_group_id(_name)


class Activations:
    """
    Stores the activations of an SFT model.
    It keeps track of:
    - the current activation's position between the layers, positive if ahead of,
      negative if behind the initial activation at position 0 (input),
    - the activations:
        - at the current position: activation in context mode, of shape [b N1 ... Nd d],
        - other position: activation in query mode, of shape [B n d],
    - the theoretical real continuous size of each activation without padding,
    - the stashes associated to each layer.
    """

    def __init__(self, input):
        self.position = 0
        self.activations = TwoWayList(lambda: None)
        self.activations[0] = input
        self.real_sizes = TwoWayList(lambda: None)
        self.real_sizes[0] = tuple(input.shape[1:-1])
        self.stashes = TwoWayList(CodecStash)
    
    def __getitem__(self, i):
        return self.activations[i]
    
    def __setitem__(self, i, value):
        self.activations[i] = value
    
    @property
    def ctx(self):
        return self[self.position]
    
    @ctx.setter
    def ctx(self, value):
        self[self.position] = value


class SFTBackbone(nn.Module):
    """SFT backbone (Scale-Free Transformer) without the stem and head."""

    def __init__(
        self,
        ctx_sizes,
        qry_sizes,
        emb_dims,
        mlp_hidden_ratios,
        n_heads,
        n_layers_codec,
        attend_to_query,
        qry_init_method,
        dropout,
        use_flash_attn,
        stop_position=None,
        n_passes=None,
        output_aux=None,
    ):
        super().__init__()
        define_backward = n_passes is None or n_passes > 1
        self.codecs = Codecs(
            define_backward,
            ctx_sizes,
            qry_sizes,
            emb_dims,
            mlp_hidden_ratios,
            n_heads,
            n_layers_codec,
            attend_to_query,
            qry_init_method,
            dropout,
            use_flash_attn,
        )
        self.stop_position = stop_position
        self.n_passes = n_passes
        self.output_aux = output_aux
        self._initialize_weights()
    
    def _initialize_weights(self):
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    @staticmethod
    def layer_idx(i0, i1):
        """Return layer index between position i0 and i1."""
        j = min(i0, i1)
        if j < 0: j = max(i0, i1) - 1
        return j
    
    @staticmethod
    def create_activations(x):
        return Activations(x)
    
    def update_activations(self, a, stop_position, n_passes, output_aux=False):
        """
        Updates a by going forth and back between a.position and stop_position (included) n_passes times.
        When using recurrent codecs, stop_position is not bounded and can be negative.
        """
        if not self.codecs.recurrent:
            assert 0 <= stop_position and stop_position <= self.codecs.n_layers
        out = []
        step = 1 if a.position < stop_position else -1
        for p in range(n_passes):
            for i in range(a.position, stop_position, step):
                i1 = i + step
                j = self.layer_idx(i, i1)
                codec = self.codecs.get(step, j)
                # Update activation i1 with activation i.
                # a[i]: context mode -> query mode
                # a[i1]: query mode -> context mode
                a[i1], a[i], real_size = codec(a[i1], a[i], a.real_sizes[i], a.stashes[j])
                if a.real_sizes[i1] is None: a.real_sizes[i1] = real_size
            a.position, stop_position = stop_position, a.position
            step = -step
            if p + 1 == n_passes or (output_aux and (p + 1) % 2 == n_passes % 2):
                out.append(a.ctx)
        return torch.cat(out[::-1], dim=0)
    
    def forward(self, x):
        """Initialize activations a from x and updates a."""
        a = self.create_activations(x)
        out = self.update_activations(a, self.stop_position, self.n_passes, self.output_aux)
        return out
    
    @property
    def num_lr_scale_groups(self):
        return self.codecs.num_lr_scale_groups
    
    def lr_scale_group_id(self, name):
        _name = name.split('.', 1)[1]
        return self.codecs.lr_scale_group_id(_name)


class SFT(nn.Module):

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
        n_layers_codec,
        attend_to_query,
        qry_init_method,
        dropout,
        use_flash_attn,
        n_passes,
        use_aux,
    ):
        super().__init__()
        assert n_passes % 2 == 1

        self.pdy = PipelineDynamics(in_size, patch_size, ctx_sizes, qry_sizes)
        self.n_layers = self.pdy.get_bottleneck_position() if len(ctx_sizes) == 1 else len(ctx_sizes)
        
        self.stem = LinearStem(in_channels, in_size, patch_size, emb_dims[0], add_sin_posemb=True)
        self.backbone = SFTBackbone(
            ctx_sizes,
            qry_sizes,
            emb_dims,
            mlp_hidden_ratios,
            n_heads,
            n_layers_codec,
            attend_to_query,
            qry_init_method,
            dropout,
            use_flash_attn,
            self.n_layers,
            n_passes,
            use_aux,
        )
        self.out_dim = emb_dims[-1]
        self.head_pre_norm = lambda: nn.LayerNorm(self.out_dim)
    
    def size_dynamics(self):
        return self.pdy.get_sizes_str(0, self.n_layers)
    
    def forward(self, x):
        x, _ = self.stem(x)
        x = self.backbone(x)
        return x
    
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
    parser = ArgumentParser("sft", add_help=False)
    parser.add_argument("--patch_size") # list[int]

    parser.add_argument("--ctx_sizes") # list[list[int]]
    parser.add_argument("--qry_sizes") # list[list[int]]

    parser.add_argument("--emb_dims") # list[int]
    parser.add_argument("--mlp_hidden_ratios") # list[int]
    parser.add_argument("--n_heads") # list[int]

    parser.add_argument("--n_layers_codec.forward") # list[list[int]]
    parser.add_argument("--n_layers_codec.backward") # list[list[int]]
    parser.add_argument("--attend_to_query.forward") # list[bool]
    parser.add_argument("--attend_to_query.backward") # list[bool]

    parser.add_argument("--qry_init_method", type=str)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--use_flash_attn", type=bool)

    parser.add_argument("--n_passes", type=int)
    parser.add_argument("--use_aux", type=bool)
    return parser


def create_encoder(config_root, config, in_channels, in_size, n_classes):
    args = get_args(config_root, (get_args_parser(), config))
    model = SFT(
        in_channels,
        in_size,
        args.patch_size,
        args.ctx_sizes,
        args.qry_sizes,
        args.emb_dims,
        args.mlp_hidden_ratios,
        args.n_heads,
        [args.n_layers_codec.forward, args.n_layers_codec.backward],
        [args.attend_to_query.forward, args.attend_to_query.backward],
        args.qry_init_method,
        args.dropout,
        args.use_flash_attn,
        args.n_passes,
        args.use_aux,
    )
    model.args = args
    return model