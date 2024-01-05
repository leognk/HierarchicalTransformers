import os
from utils import ArgumentParser, get_args
from . import unet1
from . import vit
from . import sft
from . import sft_s2s


CONFIGS_ROOT = "configs/models/encodecos"
STATIC_FILENAME = "static.yaml"


def get_static_args_parser():
    parser = ArgumentParser("Encodeco static info", add_help=False)
    parser.add_argument("--name", type=str)
    return parser


def create_encodeco(config_dir, config, in_channels, in_size):
    """
    encodeco:
        - shape: [b c N1 ... Nd] -> [b n1 ... nd d]
            where c=in_channels, N=in_size, n=in_size/model.out_patch_size, d=model.out_dim
        - output: (pred, cp)
    """
    config_root = os.path.join(CONFIGS_ROOT, config_dir)
    args = get_args(config_root, (get_static_args_parser(), STATIC_FILENAME))
    models = {
        'unet1': unet1,
        'vit': vit,
        'sft': sft,
        'sft_s2s': sft_s2s,
    }
    model = models[args.name].create_encodeco(config_root, config, in_channels, in_size)
    model.args.update(args)
    assert hasattr(model, "out_dim")
    assert hasattr(model, "out_patch_size")
    assert hasattr(model, "head_pre_norm")
    return model