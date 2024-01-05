import torch
import torch.nn as nn

from ..encoders.vit import ViTBackbone, ViT as ViT_Encoder
from ..encodecos.vit import ViT as ViT_Encodeco
from ..stems.linear import Linear as LinearStem
from ..heads.dense_predictor.linear import Linear as LinearHead
from .utils import RandomMasking, MAELoss

from einops import pack, unpack
from ..utils import get_nd_sin_posemb
import utils


class MAE_ViT(nn.Module):

    def __init__(
        self,
        in_channels,
        in_size,
        patch_size,
        emb_dim,
        n_heads,
        n_layers,
        decoder_emb_dim,
        decoder_n_heads,
        decoder_n_layers,
        mlp_hidden_ratio,
        dropout,
        use_flash_attn,
        mask_ratio,
        mask_size,
        norm_tokens,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.in_size = in_size
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.mlp_hidden_ratio = mlp_hidden_ratio
        self.n_heads = n_heads
        self.dropout = dropout
        self.use_flash_attn = use_flash_attn
        self.n_layers = n_layers
        self.norm_tokens = norm_tokens

        self.stem = LinearStem(in_channels, in_size, patch_size, emb_dim, add_sin_posemb=True)
        self.encoder = ViTBackbone(emb_dim, mlp_hidden_ratio, n_heads, dropout, None, use_flash_attn, n_layers)
        self.decoder = ViTBackbone(decoder_emb_dim, mlp_hidden_ratio, decoder_n_heads, dropout, None, use_flash_attn, decoder_n_layers)
        self.decoder_stem = nn.Linear(emb_dim, decoder_emb_dim)
        self.encoder_post_norm = nn.LayerNorm(emb_dim)
        self.decoder_post_norm = nn.LayerNorm(decoder_emb_dim)
        self.head = LinearHead(decoder_emb_dim, in_channels, patch_size, None)
        self.mae_loss = MAELoss(patch_size)
        
        psize = utils.ceil_div_it(in_size, patch_size)
        self.masking = RandomMasking(psize, mask_ratio, mask_size=mask_size)
        self.mask_token = nn.Parameter(torch.zeros(decoder_emb_dim))

        self.register_buffer(
            "decoder_posemb",
            get_nd_sin_posemb(psize, decoder_emb_dim).view(-1, decoder_emb_dim),
        )

        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.normal_(self.mask_token, mean=0, std=0.02)
    
    def forward(self, x):
        """shape: [b c N1 ... Nd] -> [1]"""
        tgts = x.clone()

        # Patchify.
        x, cp = self.stem(x) # [b c N1 ... Nd] -> [b n1 ... nd d]
        x, ps = pack([x], 'b * d') # -> [b n d]

        # Remove some tokens & encode.
        x, masks = self.masking.remove_tokens(x) # [b n d] -> [b n' d]
        x = self.encoder_post_norm(self.encoder(x))

        # Add mask tokens & decode.
        x = self.decoder_stem(x) # [b n' d] -> [b n' d']
        x = self.masking.add_mask_tokens(x, self.mask_token) # [b n' d'] -> [b n d']
        x = x + self.decoder_posemb
        x = self.decoder_post_norm(self.decoder(x))

        # Unpatch.
        [x] = unpack(x, ps, 'b * d') # -> [b n1 ... nd d']
        x = self.head(x, cp) # [b n1 ... nd d'] -> [b c N1 ... Nd]

        loss = self.mae_loss(x, tgts, masks, self.norm_tokens, patchify=True)
        return loss, "loss", x, masks
    
    def load_encoder(self, encoder):
        encoder.stem.load_state_dict(self.stem.state_dict())
        encoder.backbone.load_state_dict(self.encoder.state_dict())
    
    def load_encodeco(self, encodeco):
        encodeco.stem.load_state_dict(self.stem.state_dict())
        encodeco.backbone.load_state_dict(self.encoder.state_dict())
    
    def create_encoder(self):
        encoder = ViT_Encoder(
            self.in_channels,
            self.in_size,
            self.patch_size,
            self.emb_dim,
            self.mlp_hidden_ratio,
            self.n_heads,
            self.dropout,
            self.use_flash_attn,
            self.n_layers,
        )
        self.load_encoder(encoder)
        return encoder
    
    def create_encodeco(self):
        encodeco = ViT_Encodeco(
            self.in_channels,
            self.in_size,
            self.patch_size,
            self.emb_dim,
            self.mlp_hidden_ratio,
            self.n_heads,
            self.dropout,
            self.use_flash_attn,
            self.n_layers,
        )
        self.load_encodeco(encodeco)
        return encodeco


def get_args_parser():
    parser = utils.ArgumentParser("mae vit", add_help=False)
    parser.add_argument("--patch_size") # list[int]

    parser.add_argument("--encoder.emb_dim", type=int)
    parser.add_argument("--encoder.n_heads", type=int)
    parser.add_argument("--encoder.n_layers", type=int)

    parser.add_argument("--decoder.emb_dim", type=int)
    parser.add_argument("--decoder.n_heads", type=int)
    parser.add_argument("--decoder.n_layers", type=int)

    parser.add_argument("--mlp_hidden_ratio", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--use_flash_attn", type=bool)

    parser.add_argument("--mask_ratio", type=float)
    parser.add_argument("--mask_size") # list[int]
    parser.add_argument("--norm_tokens", type=bool)
    return parser


def create_ssl(config_root, config, in_channels, in_size):
    args = utils.get_args(config_root, (get_args_parser(), config))
    model = MAE_ViT(
        in_channels,
        in_size,
        args.patch_size,
        args.encoder.emb_dim,
        args.encoder.n_heads,
        args.encoder.n_layers,
        args.decoder.emb_dim,
        args.decoder.n_heads,
        args.decoder.n_layers,
        args.mlp_hidden_ratio,
        args.dropout,
        args.use_flash_attn,
        args.mask_ratio,
        args.mask_size,
        args.norm_tokens,
    )
    model.args = args
    return model