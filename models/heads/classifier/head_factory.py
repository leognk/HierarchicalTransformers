import os
from utils import ArgumentParser, get_args
from . import identity
from . import linear
from . import mlp


CONFIGS_ROOT = "configs/models/heads/classifier"
STATIC_FILENAME = "static.yaml"


def get_static_args_parser():
    parser = ArgumentParser("Head static info", add_help=False)
    parser.add_argument("--name", type=str)
    return parser


def create_head(config_dir, config, in_dim, out_dim, pre_norm):
    """head shape: [b * d] -> [b c'] where d=in_dim, c'=out_dim"""
    config_root = os.path.join(CONFIGS_ROOT, config_dir)
    args = get_args(config_root, (get_static_args_parser(), STATIC_FILENAME))
    models = {
        'identity': identity,
        'linear': linear,
        'mlp': mlp,
    }
    model = models[args.name].create_head(config_root, config, in_dim, out_dim, pre_norm)
    model.args.update(args)
    return model