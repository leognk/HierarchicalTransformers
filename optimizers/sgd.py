import torch
from utils import ArgumentParser, get_args
from .utils import get_param_groups


def get_args_parser():
    parser = ArgumentParser("SGD", add_help=False)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--nesterov", type=bool)
    return parser


def create_optimizer(config_root, config, model, lr, lr_decay, weight_decay):
    args = get_args(config_root, (get_args_parser(), config))
    param_groups = get_param_groups(model, lr_decay, weight_decay)
    optimizer = torch.optim.SGD(
        param_groups,
        lr=lr,
        momentum=args.momentum,
        nesterov=args.nesterov,
    )
    optimizer.args = args
    return optimizer