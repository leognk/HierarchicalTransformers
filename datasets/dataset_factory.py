import os
import utils
from .subset import ClassificationSubset
from .cifar import CIFAR10, CIFAR100
from .imagenet import ImageNet
from .mini_imagenet import MiniImageNet
from .mini_imagenet_plus import MiniImageNetPlus


CONFIGS_ROOT = "configs/datasets"
STATIC_FILENAME = "static.yaml"


def get_static_args_parser():
    parser = utils.ArgumentParser("Dataset static info", add_help=False)
    parser.add_argument("--name", type=str)
    parser.add_argument("--data_root") # str
    return parser


def get_args_parser():
    parser = utils.ArgumentParser("Dataset specific info", add_help=False)
    parser.add_argument("--classes") # int, list[int] or list[str], optional
    parser.add_argument("--n_samples") # dict[int], optional
    parser.add_argument("--classes_seed", type=int) # optional
    parser.add_argument("--samples_seed") # dict[int], optional
    return parser


def create_dataset(config_dir, config, train, transform=None, exclude_ids=None):
    config_root = os.path.join(CONFIGS_ROOT, config_dir)
    args = utils.get_args(config_root, (get_static_args_parser(), STATIC_FILENAME), (get_args_parser(), config))
    datasets = {
        'cifar10': CIFAR10,
        'cifar100': CIFAR100,
        'imagenet': ImageNet,
        'mini_imagenet': MiniImageNet,
        'mini_imagenet_plus': MiniImageNetPlus,
    }
    kwargs = {'exclude_ids': exclude_ids} if exclude_ids else {}
    dataset = datasets[args.name](args.data_root, train, transform, **kwargs)
    n_samples = args.n_samples[dataset.split]
    samples_seed = args.samples_seed[dataset.split]
    if args.classes or n_samples:
        dataset = ClassificationSubset(dataset, args.classes, n_samples, args.classes_seed, samples_seed)
    dataset.args = args
    dataset.config_root = config_root
    return dataset


def dataset_str(d):
    head = f"Dataset {d.name}"
    body = '\n'.join([
        f"Split: {d.split}",
        f"Nb samples: {utils.pretty_number(len(d), 2)}",
        f"Nb classes: {utils.pretty_number(d.n_classes, 0)}",
        f"Avg nb samples/class: {utils.pretty_number(d.avg_n_samples_by_class, 2)}",
    ])
    return utils.join_head_body(head, body)