import argparse
import os
from logger import Run
import utils


def get_args_parser():
    parser = argparse.ArgumentParser("Remove useless checkpoints", add_help=False)
    parser.add_argument(
        "--root",
        default="test1",
        type=str,
    )
    # Keep ckpts from epochs multiples of this value.
    parser.add_argument(
        "--keep_epoch_freq",
        default=None,
        type=int,
    )
    return parser


def rm_all_useless_ckpts(root, keep_epoch_freq, verbose=False):
    """
    Iterate through all subdirs in given root (a subdir of runs root)
    and remove useless ckpts when found.
    """
    runs_root = utils.get_runs_root()
    root = os.path.join(runs_root, root)
    for dir, _, _ in os.walk(root):
        Run.rm_useless_ckpts(dir, keep_epoch_freq, verbose)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    rm_all_useless_ckpts(args.root, args.keep_epoch_freq, verbose=True)