from ..type import ModelType
import os
from utils import ArgumentParser, get_args
from . import mae_unet1
from . import mae_vit
from . import mae_sft
from . import mae_sft_s2s
from . import hmae_sft


CONFIGS_ROOT = "configs/models/ssl"
STATIC_FILENAME = "static.yaml"


def get_static_args_parser():
    parser = ArgumentParser("SSL static info", add_help=False)
    parser.add_argument("--name", type=str)
    return parser


def create_ssl(config_dir, config, in_channels, in_size):
    """
    SSL model:
        - I/O: img -> loss, loss_name, ... (optional other tensors)
        - SSL model = arbitrary model (no constraint such as encoder + head)
        - shape: [b c N1 ... Nd] -> [1] (loss)
            where c=in_channels, N=in_size
        - defines create_encoder/create_encodeco
    """
    config_root = os.path.join(CONFIGS_ROOT, config_dir)
    args = get_args(config_root, (get_static_args_parser(), STATIC_FILENAME))
    models = {
        'mae_unet1': mae_unet1,
        'mae_vit': mae_vit,
        'mae_sft': mae_sft,
        'mae_sft_s2s': mae_sft_s2s,
        'hmae_sft': hmae_sft,
    }
    model = models[args.name].create_ssl(config_root, config, in_channels, in_size)
    model.args.update(args)
    model.type = ModelType.SSL
    return model