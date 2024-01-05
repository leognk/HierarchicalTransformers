import os
from utils import ArgumentParser, get_args
from . import enc1
from . import enc2
from . import unet1_enc
from . import vit
from .sft import sft
from . import sft_s2s
from . import hvit
from . import hvit2
from . import hvit3
from . import hvit4
from . import swin
from . import swin2
from . import attn_pool
from . import attn_pool2
from . import metaformer
from . import fast_ds


CONFIGS_ROOT = "configs/models/encoders"
STATIC_FILENAME = "static.yaml"


def get_static_args_parser():
    parser = ArgumentParser("Encoder static info", add_help=False)
    parser.add_argument("--name", type=str)
    return parser


def create_encoder(config_dir, config, in_channels, in_size, n_classes):
    """
    encoder shape: [b c N1 ... Nd] -> [b n1 ... nd d] or [b d]
        where c=in_channels, N=in_size, d=model.out_dim
    """
    config_root = os.path.join(CONFIGS_ROOT, config_dir)
    args = get_args(config_root, (get_static_args_parser(), STATIC_FILENAME))
    models = {
        'enc1': enc1,
        'enc2': enc2,
        'unet1_enc': unet1_enc,
        'vit': vit,
        'sft': sft,
        'sft_s2s': sft_s2s,
        'hvit': hvit,
        'hvit2': hvit2,
        'hvit3': hvit3,
        'hvit4': hvit4,
        'swin': swin,
        'swin2': swin2,
        'attn_pool': attn_pool,
        'attn_pool2': attn_pool2,
        'metaformer': metaformer,
        'fast_ds': fast_ds,
    }
    model = models[args.name].create_encoder(config_root, config, in_channels, in_size, n_classes)
    model.args.update(args)
    assert hasattr(model, "out_dim")
    assert hasattr(model, "head_pre_norm")
    return model