import os
from utils import ArgumentParser, get_args
from . import sgd
from . import adamw
from . import adam
from . import lamb


CONFIGS_ROOT = "configs/optimizers"
STATIC_FILENAME = "static.yaml"


def get_static_args_parser():
    parser = ArgumentParser("Optimizer static info", add_help=False)
    parser.add_argument("--name", type=str)
    return parser


def create_optimizer(config_dir, config, model, lr, lr_decay, weight_decay):
    config_root = os.path.join(CONFIGS_ROOT, config_dir)
    args = get_args(config_root, (get_static_args_parser(), STATIC_FILENAME))
    optimizers = {
        'SGD': sgd,
        'AdamW': adamw,
        'Adam': adam,
        'Lamb': lamb,
    }
    optimizer = optimizers[args.name].create_optimizer(
        config_root,
        config,
        model,
        lr,
        lr_decay,
        weight_decay,
    )
    optimizer.args.update(args)
    return optimizer