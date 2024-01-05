import torch
import torch.nn as nn

from ..encoders.unet1_enc import UNet1_Encoder
from ..encodecos.unet1 import UNet1
from ..heads.dense_predictor.linear import Linear as LinearHead
from .utils import RandomMasking, MAELoss

from ..stems import get_patchify
from ..heads.dense_predictor import get_unpatch
from ..utils import flat_dim
import utils


class MAEUNet1(nn.Module):
    """Only for 2D data."""

    def __init__(self, in_channels, in_size, mask_patch_size, mask_ratio, norm_tokens):
        super().__init__()

        self.in_channels = in_channels
        self.in_size = in_size
        self.norm_tokens = norm_tokens

        self.backbone = UNet1(in_channels, in_size)
        self.head = LinearHead(self.backbone.out_dim, in_channels, self.backbone.out_patch_size, self.backbone.head_pre_norm)
        self.mae_loss = MAELoss(mask_patch_size)

        psize = utils.ceil_div_it(in_size, mask_patch_size)
        self.masking = RandomMasking(psize, mask_ratio)
        patch_numel = flat_dim(in_channels, mask_patch_size)
        self.mask_token = nn.Parameter(torch.zeros(patch_numel))

        self.patchify = get_patchify(mask_patch_size)
        self.unpatch = get_unpatch(mask_patch_size)

        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.normal_(self.mask_token, mean=0, std=0.02)
    
    def random_masking(self, x):
        """Mask some portion of x with patches."""
        x, cp = self.patchify(x) # [b c N1 ... Nd] -> [b n1 ... nd d]
        x, masks = self.masking.replace_with_mask_tokens(x, self.mask_token)
        x = self.unpatch(x, cp) # [b n1 ... nd d] -> [b c N1 ... Nd]
        return x, masks
    
    def forward(self, x):
        """shape: [b c N1 ... Nd] -> [1]"""
        tgts = x.clone()
        x, masks = self.random_masking(x)
        x, cp = self.backbone(x) # [b c N1 ... Nd] -> [b N1 ... Nd d]
        x = self.head(x, cp) # -> [b c N1 ... Nd]
        loss = self.mae_loss(x, tgts, masks, self.norm_tokens, patchify=True)
        return loss, "loss", x, masks
    
    def load_encoder(self, encoder):
        encoder.load_state_dict(self.backbone.state_dict(), strict=False)
    
    def load_encodeco(self, encodeco):
        encodeco.load_state_dict(self.backbone.state_dict())
    
    def create_encoder(self):
        encoder = UNet1_Encoder(self.in_channels, self.in_size)
        self.load_encoder(encoder)
        return encoder
    
    def create_encodeco(self):
        encodeco = UNet1(self.in_channels, self.in_size)
        self.load_encodeco(encodeco)
        return encodeco


def get_args_parser():
    parser = utils.ArgumentParser("mae_unet1", add_help=False)
    parser.add_argument("--mask_patch_size") # list[int]
    parser.add_argument("--mask_ratio", type=float)
    parser.add_argument("--norm_tokens", type=bool)
    return parser


def create_ssl(config_root, config, in_channels, in_size):
    args = utils.get_args(config_root, (get_args_parser(), config))
    model = MAEUNet1(in_channels, in_size, args.mask_patch_size, args.mask_ratio, args.norm_tokens)
    model.args = args
    return model