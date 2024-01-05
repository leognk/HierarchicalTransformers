import os
import utils
from .transforms import create_augmentation


TRANSFORMS_DIRNAME = "transforms"


def get_args_parser():
    parser = utils.ArgumentParser("Transform", add_help=False)
    # img_size & interpolation are responsible for resizing.
    # It can be disabled only if the dataset images have a fixed size.
    parser.add_argument("--img_size") # int or list[int], optional: if list[int], shape=[h w]
    parser.add_argument("--interpolation", type=str) # optional
    parser.add_argument("--crop_pct", type=float) # optional
    parser.add_argument("--crop_scale") # list[float], optional
    parser.add_argument("--crop_ratio") # list[float], optional
    parser.add_argument("--hflip", type=float) # optional
    parser.add_argument("--vflip", type=float) # optional
    parser.add_argument("--aa_config", type=str) # optional
    parser.add_argument("--aa_interpolation", type=str) # optional
    parser.add_argument("--color_jitter") # float or list[float], optional
    parser.add_argument("--re_prob", type=float) # optional
    parser.add_argument("--re_value") # int, list[int] or str, optional
    return parser


def create_transform(dataset, config):
    """
    Args:
        - dataset (Dataset)
        - config (str)
    Returns:
        - transform:
            - input: PIL Image of arbitrary size
            - output: normalized PyTorch Tensor of size transform.img_size
    """
    config_root = os.path.join(dataset.config_root, TRANSFORMS_DIRNAME)
    args = utils.get_args(config_root, (get_args_parser(), config))

    assert utils.all_none_or_not(args.img_size, args.interpolation)
    img_size = args.img_size
    if img_size is None:
        assert dataset.height is not None and dataset.width is not None, "default image size is undefined"
        img_size = dataset.height, dataset.width
    elif isinstance(img_size, int):
        img_size = (img_size, img_size)

    transform = create_augmentation(
        img_size,
        dataset.mean,
        dataset.std,
        args.interpolation,
        args.crop_pct,
        args.crop_scale,
        args.crop_ratio,
        args.hflip,
        args.vflip,
        args.aa_config,
        args.aa_interpolation,
        args.color_jitter,
        args.re_prob,
        args.re_value,
    )
    transform.args = args
    transform.channels = dataset.channels
    transform.img_size = img_size
    return transform


def transform_str(name, t):
    h, w = t.img_size
    body = '\n'.join([
        f"Image [C H W]: [{t.channels} {h} {w}]",
        str(t),
    ])
    return utils.join_head_body(name, body)