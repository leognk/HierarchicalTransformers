import os
import json
import jsonargparse
import yaml
import csv
import torch
import torch.distributed
import torch.utils.data
import numpy as np
import random
import math
import collections
import datetime
import builtins


#################### FILE LOADING ####################


def load_json(filepath):
    with open(filepath, 'rb') as f:
        js = json.load(f)
    return js


def save_json(js, filepath, indent=None):
    with open(filepath, 'w') as f:
        json.dump(js, f, indent=indent)


def load_yaml(filepath):
    with open(filepath, 'rb') as f:
        ym = yaml.safe_load(f)
    return ym


def load_csv(filepath, skip_header=True):
    with open(filepath) as f:
        reader = csv.reader(f)
        if skip_header: next(reader, None)
        data = list(reader)
    return data


#################### ARGUMENT PARSING ####################


HOST_PATH = "configs/host.yaml"
RUNS_PATH = "configs/runs.yaml"


def get_host():
    return load_yaml(HOST_PATH)["host"]


def get_runs_root():
    return load_yaml(RUNS_PATH)["runs_root"][get_host()]


class ArgumentParser(jsonargparse.ArgumentParser):
    """
    Can parse a config file.
    For options with different values for each possible host,
    select the value for the current host.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.host = get_host()
        self.add_argument("--config", action=jsonargparse.ActionConfigFile)
    
    @staticmethod
    def _select_option(args, option):
        for k, v in vars(args).items():
            if isinstance(v, jsonargparse.Namespace):
                ArgumentParser._select_option(v, option) # recursively iterate over v
            if isinstance(v, dict) and option in v:
                setattr(args, k, v[option])

    def parse_args(self, *args, **kwargs):
        res = super().parse_args(*args, **kwargs)
        self._select_option(res, self.host)
        res.pop("config")
        return res
    
    def parse_config(self, config):
        return self.parse_args(["--config", config])


def args_str(args):
    cfg = args.clone()
    for k, v in args.items():
        if v is None: cfg.pop(k)
    return join_head_body("Config", yaml.safe_dump(cfg.as_dict(), sort_keys=False))


def merge_dicts(*dicts):
    res = {}
    for d in dicts:
        res.update(d)
    return res


def merge_args(*args):
    res = jsonargparse.Namespace()
    for a in args:
        res.update(a)
    return res


def get_args(config_root, *parsers_configs):
    """
    Args:
        - config_root (str)
        - parsers_configs (pair[ArgumentParser, str], ...)
    Returns:
        - args (Namespace)
    """
    lst_args = [parser.parse_config(os.path.join(config_root, config)) for parser, config in parsers_configs]
    return merge_args(*lst_args)


#################### DISTRIBUTED ####################


class DDP:

    backend = 'nccl'

    def __init__(self):
        if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
            torch.distributed.init_process_group(backend=self.backend)
            self._print_in_master()
            world_size_str = quantity_str(self.world_size, "process", "processes")
            print(f"Initialized distributed mode with {world_size_str}.")
        else:
            print("Not using distributed mode.")
    
    def _print_in_master(self):
        """Disable printing when not in master."""
        builtin_print = builtins.print

        def print(*args, **kwargs):
            force = kwargs.pop("force", False)
            if self.is_master or force:
                builtin_print(*args, **kwargs)
        
        builtins.print = print
    
    @property
    def use_ddp(self):
        return torch.distributed.is_initialized()
    
    @property
    def rank(self):
        return torch.distributed.get_rank() if self.use_ddp else 0
    
    @property
    def local_rank(self):
        return int(os.environ["LOCAL_RANK"]) if self.use_ddp else 0
    
    @property
    def world_size(self):
        return torch.distributed.get_world_size() if self.use_ddp else 1
    
    @property
    def is_master(self):
        return self.rank == 0


class DistributedSamplerNoDuplicate(torch.utils.data.DistributedSampler):
    """
    A distributed sampler that doesn't add duplicates.
    from: https://github.com/pytorch/pytorch/issues/25162#issuecomment-1227647626
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # some ranks may have less samples, that's fine
            if self.rank >= len(self.dataset) % self.num_replicas:
                self.num_samples -= 1
            self.total_size = len(self.dataset)


#################### MISCELLANEOUS ####################


def all_true_or_false(*args):
    return all(args) or not any(args)


def all_none_or_not(*args):
    return all_true_or_false(*[x is None for x in args])


def count_true(*args):
    return sum(bool(x) for x in args)


def get_timedelta(dt):
    return datetime.timedelta(seconds=round(dt))


def get_timedeltas(*args):
    return (get_timedelta(dt) for dt in args)


def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_amp_dtype(use_amp):
    if not use_amp: return None
    if torch.cuda.is_bf16_supported(): return torch.bfloat16
    return torch.float16


def count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def freeze_params(module):
    for p in module.parameters():
        p.requires_grad = False


class LearningRateSetter:

    def __init__(self, lr, use_decay, warmup_epochs, decay_epochs, min_lr):
        self.lr = lr
        self.use_decay = use_decay
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.min_lr = min_lr
    
    def _get_lr(self, epoch):
        if epoch < self.warmup_epochs:
            return self.lr * epoch / self.warmup_epochs
        if epoch > self.decay_epochs:
            return self.min_lr
        decay_ratio = (epoch - self.warmup_epochs) / (self.decay_epochs - self.warmup_epochs)
        alpha = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return alpha * self.lr + (1 - alpha) * self.min_lr
    
    def __call__(self, optimizer, epoch):
        if not self.use_decay: return self.lr
        lr = self._get_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
            if "lr_scale" in param_group:
                param_group["lr"] *= param_group["lr_scale"]
        return lr


def filter_state_dict(state_dict, module_name):
    res = collections.OrderedDict()
    for k, v in state_dict.items():
        ks = k.split('.', 1)
        if len(ks) == 2 and ks[0] == module_name:
            res[ks[1]] = v
    return res


def accuracy(preds, targets, topk=(1,)):
    """
    Computes the accuracy over the top k predictions for the specified values of k.
    Args:
        - preds (Tensor): shape: [b c]
        - targets (Tensor): shape: [b]
        - topk (tuple)
    Returns:
        - accuracies (list[float])
    """
    assert preds.dim() == 2 and targets.dim() == 1
    b, c = preds.shape
    maxk = min(max(topk), c)
    _, preds = torch.topk(preds, maxk, dim=1, largest=True, sorted=True) # [b k]
    corrects = preds.T == targets # [k b] x [b] -> [k b]
    return [100 * corrects[:min(k, maxk)].flatten().float().sum().item() / b for k in topk]


def repeat_tensor(x, n):
    """
    Repeat x without copy.
    x: [b ...] -> [(n b) ...]
    """
    return x.expand(n, *x.shape).reshape(-1, *x.shape[1:])


def ravel_multi_index(multi_index, dims):
    """
    Equivalent of np.ravel_multi_index for pytorch.
    multi_index: [d n]
    dims: [d]
    out: [n]
    """
    return torch.sum(torch.stack([
        multi_index[i] * product(dims[i+1:]) for i in range(len(dims))
    ]), dim=0)


def filter_dict(dct, keys):
    return {k: v for k, v in dct.items() if k not in keys}


#################### OPERATIONS ####################


def clamp(x, a, b):
    return max(a, min(x, b))


def product(it, start=1):
    res = start
    for n in it: res *= n
    return res


def sign(n):
    if n > 0: return 1
    if n < 0: return -1
    return 0


def ceil_div(x, y):
    return int(math.ceil(round(x / y, 8))) # round to eliminate precision errors


def add_it(it1, it2):
    return it1.__class__(x + y for x, y in zip(it1, it2))


def subtract_it(it1, it2):
    return it1.__class__(x - y for x, y in zip(it1, it2))


def mult_it(it1, it2):
    return it1.__class__(x * y for x, y in zip(it1, it2))


def div_it(it1, it2):
    return it1.__class__(x / y for x, y in zip(it1, it2))


def floor_div_it(it1, it2):
    return it1.__class__(int(x // y) for x, y in zip(it1, it2))


def ceil_div_it(it1, it2):
    return it1.__class__(ceil_div(x, y) for x, y in zip(it1, it2))


def max_it(it1, it2):
    return it1.__class__(max(x, y) for x, y in zip(it1, it2))


def min_it(it1, it2):
    return it1.__class__(min(x, y) for x, y in zip(it1, it2))


def less_than_it(it1, it2):
    for x, y in zip(it1, it2):
        if not (x < y): return False
    return True


#################### STRING REPRESENTATION ####################


def pretty_number(n, decimals):
    big_suffixes = ['', 'K', 'M', 'G', 'T', 'P']
    small_suffixes = ['', 'm', 'u', 'n', 'p', 'f']
    if n == 0:
        suffix = ''
    elif abs(n) >= 1:
        scale = 0
        while abs(round(n, decimals)) >= 1000:
            n /= 1000
            scale += 1
            if scale == len(big_suffixes) - 1: break
        suffix = big_suffixes[scale]
    else:
        scale = 0
        while abs(round(n, decimals)) < 1:
            n *= 1000
            scale += 1
            if scale == len(small_suffixes) - 1: break
        suffix = small_suffixes[scale]
    if int(n) == n: decimals = 0
    return f"{n:.{decimals}f}{suffix}"


def join_head_body(head, body, indent=4):
    lines = [head] + [' ' * indent + line for line in body.splitlines()]
    return '\n'.join(lines)


def max_line(body):
    return max(len(line) for line in body.split('\n'))


def quantity_str(n, singular_unit, plural_unit, fmt=""):
    unit = plural_unit if abs(n) > 1 else singular_unit
    return f"{format(n, fmt)} {unit}"


def dataloader_str(name, dataloader):
    n_batches = len(dataloader)
    batch_size = dataloader.batch_size
    n_samples_guess = n_batches * batch_size

    n_batches = pretty_number(n_batches, 2)
    batch_size = quantity_str(batch_size, "sample", "samples")
    n_samples_guess = pretty_number(n_samples_guess, 2)

    body = f"{n_batches} batches x {batch_size} ~ {n_samples_guess} samples"
    return join_head_body(name, body)