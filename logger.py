import os
import path
import glob
import shutil
import re
import math
import time
import datetime
import torch
import torch.distributed
import wandb
from collections import deque, OrderedDict, defaultdict
import utils
from datasets import dataset_str, transform_str


class Date:

    @staticmethod
    def now():
        return datetime.datetime.now()
    
    @staticmethod
    def to_str(date):
        return date.strftime("%a %d %b, %H:%M:%S")
    
    @staticmethod
    def to_tuple(date):
        return tuple(date.timetuple())[:6]


class Timer:

    def __init__(self, start=False):
        self.dt = 0
        self.start_time = None
        if start: self.start()
    
    def start(self):
        if self.start_time is None:
            self.start_time = time.time()
    
    def stop(self):
        self.dt += self.run()
        self.start_time = None
    
    def run(self):
        if self.start_time is None: return 0
        return time.time() - self.start_time
    
    def rundelta(self):
        return utils.get_timedelta(self.run())
    
    def time(self):
        return self.dt + self.run()
    
    def timedelta(self):
        return utils.get_timedelta(self.time())
    
    def state_dict(self):
        return {"dt": self.time()}
    
    def load_state_dict(self, state_dict):
        self.dt = state_dict["dt"]


class TimeProgress:

    EPOCHS_STR = "epochs"

    def __init__(self, n_iters, steps_per_iter, iter_name):
        self.n_iters = n_iters
        self.steps_per_iter = steps_per_iter
        self.iters_n_steps = [steps_per_iter] * n_iters
        self.n_steps = sum(self.iters_n_steps)
        self.iter_name = iter_name
        self.timer = Timer()
        self.iter = 0
        self.step = 0
        self.local_step = 0
    
    def resume(self):
        self.timer.start()
    
    def pause(self):
        self.timer.stop()
    
    def start_iter(self):
        self.local_timer = Timer(start=True)
        self.local_step = 0
    
    def end_iter(self):
        self.iter += 1
        self.local_step = 0
    
    def end_step(self):
        self.step += 1
        self.local_step += 1
    
    def time(self):
        return self.timer.time()
    
    def timedelta(self):
        return self.timer.timedelta()
    
    @staticmethod
    def _time_progress(dt, step, n_steps, local_dt=None):
        total_dt = dt / step * n_steps if step != 0 else 0
        remaining_dt = total_dt - dt
        if local_dt is not None:
            dt = local_dt
            total_dt = dt + remaining_dt
        return dt, remaining_dt, total_dt

    @staticmethod
    def _get_finish_date(remaining_dt):
        current_time = Date.now()
        finish_time = current_time + datetime.timedelta(seconds=remaining_dt)
        current_date = Date.to_str(current_time)
        finish_date = Date.to_str(finish_time)
        return current_date, finish_date
    
    @staticmethod
    def _pretty_time_progress(dt, remaining_dt, total_dt):
        current_date, finish_date = TimeProgress._get_finish_date(remaining_dt)
        dt, remaining_dt, total_dt = utils.get_timedeltas(dt, remaining_dt, total_dt)
        time_progress = f"{remaining_dt} -> {dt} / {total_dt}"
        finish_date = f"{current_date} -> {finish_date}"
        return time_progress, finish_date
    
    @staticmethod
    def _minimal_pretty_time_progress(dt, remaining_dt, total_dt):
        _, finish_date = TimeProgress._get_finish_date(remaining_dt)
        _, remaining_dt, _ = utils.get_timedeltas(dt, remaining_dt, total_dt)
        return str(remaining_dt), finish_date
    
    def time_progress(self, local=False, n_iters=None):
        dt = self.timer.time()
        if local:
            local_dt = self.local_timer.time()
            if n_iters is None: n_iters = self.iter + 1
        else:
            local_dt = None
            if n_iters is None: n_iters = self.n_iters
        n_steps = sum(self.iters_n_steps[:n_iters])
        return self._time_progress(dt, self.step, n_steps, local_dt)
    
    def pretty_time_progress(self, local=False):
        return self._pretty_time_progress(*self.time_progress(local))
    
    @staticmethod
    def _pretty_iter_progress(iter, n_iters, percentage, iter_name):
        iter_progress = f"{iter}/{n_iters} {iter_name}"
        percentage = f"{percentage:.2f}%"
        return f"{iter_progress} ({percentage})"
    
    def iter_progress(self, local=False):
        if local:
            iter = self.local_step
            n_iters = self.steps_per_iter
            percentage = 100 * iter / n_iters
            iter_name = "steps"
        else:
            iter = self.iter
            n_iters = self.n_iters
            percentage = 100 * self.step / self.n_steps if self.n_steps != 0 else 100
            iter_name = self.iter_name
        return iter, n_iters, percentage, iter_name
    
    def pretty_iter_progress(self, local=False):
        return self._pretty_iter_progress(*self.iter_progress(local))
    
    @staticmethod
    def _get_str(time_progress, finish_date, iter_progress):
        return '\n'.join([finish_date, time_progress, iter_progress])
    
    def get_str(self, local=False):
        return self._get_str(*self.pretty_time_progress(local), self.pretty_iter_progress(local))
    
    def state_dict(self):
        return {
            "timer": self.timer.state_dict(),
            "iter": self.iter,
            "step": self.step,
            "iters_n_steps": self.iters_n_steps[:self.iter],
        }
    
    def load_state_dict(self, state_dict):
        self.timer.load_state_dict(state_dict["timer"])
        self.iter = state_dict["iter"]
        self.step = state_dict["step"]
        iters_n_steps = state_dict["iters_n_steps"]
        self.iters_n_steps[:len(iters_n_steps)] = iters_n_steps
        self.n_steps = sum(self.iters_n_steps)


class GlobalTimeProgress:

    def __init__(self):
        self.tps = []
        self.lst_iter_epochs = []
    
    def register(self, tp, iter_epochs):
        self.tps.append(tp)
        self.lst_iter_epochs.append(iter_epochs)
        if tp.iter_name == TimeProgress.EPOCHS_STR:
            self.epoch_tp = tp
    
    @property
    def epoch(self):
        return self.epoch_tp.iter + self.epoch_tp.local_step / self.epoch_tp.steps_per_iter
    
    @property
    def n_epochs(self):
        return self.epoch_tp.n_iters
    
    def n_iters(self, epoch):
        res = []
        for iter_epochs, tp in zip(self.lst_iter_epochs, self.tps):
            n = epoch + 1 if tp.iter_name == TimeProgress.EPOCHS_STR else epoch
            res.append(sum(iter_epochs[:n]))
        return res
    
    def time(self):
        return sum(tp.time() for tp in self.tps)

    def timedelta(self):
        return sum((tp.timedelta() for tp in self.tps), start=datetime.timedelta())

    def time_progress(self, epoch=None):
        if epoch is not None: n_iters = self.n_iters(epoch)
        res = (0, 0, 0)
        for i, tp in enumerate(self.tps):
            n = n_iters[i] if epoch is not None else None
            res = utils.add_it(res, tp.time_progress(n_iters=n))
        return res
    
    def pretty_time_progress(self):
        return TimeProgress._pretty_time_progress(*self.time_progress())
    
    def minimal_pretty_time_progress(self, epoch=None):
        return TimeProgress._minimal_pretty_time_progress(*self.time_progress(epoch))
    
    def pretty_iter_progress(self):
        res = []
        for tp in self.tps:
            res.append(tp.pretty_iter_progress())
        return " | ".join(res)
    
    def get_str(self):
        return TimeProgress._get_str(*self.pretty_time_progress(), self.pretty_iter_progress())


class WandB:

    PROJECT = "SFT"
    LOCAL_ROOT = "wandb"
    RUN_ID_FILENAME = "run_id.json"
    MAX_N_LOGS = 10000

    def __init__(self, run_name, run_path):
        self.loaded_state_dict = False
        self.logs = []
        self.summary = {}

        # Delete the previously stopped run if it exists to make a fresh restart
        # and later load WandB from a checkpoint.
        run_id_path = os.path.join(run_path, self.RUN_ID_FILENAME)
        if path.Path(run_id_path).exists():
            run_id = utils.load_json(run_id_path)["run_id"]
            self._delete_run_if_exists(run_id)

        wandb.init(project=self.PROJECT, name=run_name)
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

        os.makedirs(run_path, exist_ok=True)
        utils.save_json({"run_id": wandb.run.id}, run_id_path)
    
    def _delete_run_if_exists(self, run_id):
        # Delete local wandb run folder.
        if path.Path(self.LOCAL_ROOT).exists():
            for entry in (os.listdir(self.LOCAL_ROOT)):
                if re.match(rf'.+-{run_id}$', entry):
                    shutil.rmtree(os.path.join(self.LOCAL_ROOT, entry))
        
        # Delete from wandb server.
        api = wandb.Api()
        try: run = api.run(f"{self.PROJECT}/{run_id}")
        except wandb.CommError: return
        run.delete(delete_artifacts=True)
    
    def finish(self):
        wandb.finish()
    
    def update_config(self, config):
        wandb.config.update(config)
    
    def _log_dict(self, log):
        self.logs.append(log)
        wandb.log(log)
    
    def log_summary(self, k, v):
        self.summary[k] = v
        wandb.run.summary[k] = v
    
    def log(self, epoch, metrics, best_metric_name=None, best_metric=None, **extra_info):
        log = {
            "epoch": epoch,
            **{f"{metrics.name}/{k}": v for k, v in metrics.items()},
        }
        self._log_dict(log)
        assert utils.all_none_or_not(best_metric_name, best_metric)
        if best_metric:
            k = f"best_{metrics.name}_{best_metric_name}"
            self.log_summary(k, best_metric)
        if extra_info:
            for k, v in extra_info.items():
                self.log_summary(k, v)
    
    def state_dict(self):
        return {
            "logs": self.logs,
            "summary": self.summary,
        }
    
    def load_state_dict(self, state_dict):
        assert not self.loaded_state_dict
        for log in state_dict["logs"]:
            self._log_dict(log)
        for k, v in state_dict["summary"].items():
            self.log_summary(k, v)
        self.loaded_state_dict = True


class Run:
    """Manages run name, checkpoints and WandB."""

    def __init__(self, name, resume_ckpt, save_ckpt, ckpt_keep_freq, use_wandb):
        # Set name and path.
        root = utils.get_runs_root()
        if not name: name = self._default_name(root)
        self.name = name
        self.path = os.path.join(root, self.name)
        if save_ckpt: os.makedirs(self.path, exist_ok=True)
        self.ckpt_keep_freq = ckpt_keep_freq

        # Look for the checkpoint path.
        if not resume_ckpt:
            self.resume_ckpt = self.find_last_ckpt(self.path)
        else:
            ckpt_path = os.path.join(self.path, resume_ckpt)
            if not path.Path(ckpt_path).exists():
                raise Exception(f"Could not find checkpoint at: {ckpt_path}")
            self.resume_ckpt = resume_ckpt
        
        # Init wandb.
        self.wandb = WandB(self.name, self.path) if use_wandb else None

        if self.resume_ckpt:
            print(f'-\nResume run "{name}" from checkpoint {self.resume_ckpt}')
        else:
            print(f'-\nStart new run "{name}"')

    @staticmethod
    def _default_name(root):
        current_time = Date.now()
        current_date = current_time.strftime("%m-%d-%H:%M:%S")
        same_names = glob.glob(os.path.join(root, current_date + '*'))
        # If the name already exists, add an id at the end.
        if same_names:
            # Find the maximum id among all the names of the form "{date}-{id}".
            max_id = 1
            for name in same_names:
                m = re.findall(r'-(\d+)$', name)
                if m: max_id = max(max_id, int(m[0]))
            return f"{current_date}-{max_id + 1}"
        return current_date
    
    @staticmethod
    def ckpt_name(epoch):
        return f"ckpt-{epoch}.pt" if epoch is not None else None
    
    @staticmethod
    def extract_ckpt_epoch(ckpt_name):
        m = re.findall(r'ckpt-(\d+)\.pt$', ckpt_name)
        return int(m[0]) if m else None
    
    @staticmethod
    def find_last_ckpt(run_path):
        if not path.Path(run_path).exists(): return None
        last_epoch = None
        for entry in os.listdir(run_path):
            epoch = Run.extract_ckpt_epoch(entry)
            if epoch is None: continue
            if last_epoch is None or epoch > last_epoch: last_epoch = epoch
        return Run.ckpt_name(last_epoch)
    
    @staticmethod
    def is_ckpt_file(ckpt_name):
        epoch = Run.extract_ckpt_epoch(ckpt_name)
        return epoch is not None

    @staticmethod
    def load_ckpt(ckpt_name, root=None, map_location=None):
        """
        Utility function for loading ckpt file.
        If ckpt_name is:
            - None: return None
            - a ckpt file name: return the ckpt with this name
            - a run name: return the last ckpt of the run
        """
        if not ckpt_name: return None, None
        if not root: root = utils.get_runs_root()
        if not Run.is_ckpt_file(ckpt_name):
            last_ckpt = Run.find_last_ckpt(os.path.join(root, ckpt_name))
            assert last_ckpt
            ckpt_name = os.path.join(ckpt_name, last_ckpt)
        ckpt_path = os.path.join(root, ckpt_name)
        return torch.load(ckpt_path, map_location=map_location), ckpt_path
    
    @staticmethod
    def rm_useless_ckpts(run_path, keep_epoch_freq, verbose=False):
        """
        If run_path is not an actual run folder, just return.
        If keep_epoch_freq is None, remove all ckpts except the last & best.
        """

        # Try to find the last ckpt.
        last_ckpt_name = Run.find_last_ckpt(run_path)
        if last_ckpt_name is None: return
        last_epoch = Run.extract_ckpt_epoch(last_ckpt_name)
        last_ckpt_path = os.path.join(run_path, last_ckpt_name)
        last_ckpt = torch.load(last_ckpt_path, map_location='cpu')['logger']

        # Find the best ckpt.
        if last_ckpt['eval']['best_metrics']:
            best_epoch = last_ckpt['eval']['best_metrics']['metrics']['metrics']['epoch']
        else:
            best_epoch = None
        best_ckpt_name = Run.ckpt_name(best_epoch)

        # Remove all ckpts except the last, best, and multiples of keep_epoch_freq.
        count = 0
        kept_best = False
        for entry in os.listdir(run_path):
            epoch = Run.extract_ckpt_epoch(entry)
            if epoch is not None and epoch == best_epoch:
                kept_best = True
            if epoch is None or epoch in [last_epoch, best_epoch] or (keep_epoch_freq and epoch % keep_epoch_freq == 0):
                continue
            os.remove(os.path.join(run_path, entry))
            count += 1
        
        # Print info
        if verbose:
            count_str = utils.quantity_str(count, "ckpt", "ckpts")
            print()
            print(f"run: {run_path}")
            print(f"removed {count_str}")
            print(f"kept last ckpt: {last_ckpt_name}")
            if kept_best:
                print(f"kept best ckpt: {best_ckpt_name}")
            if keep_epoch_freq:
                print(f"kept multiples of {keep_epoch_freq}")
    
    def _compare_args(self, const_args, prev_args):
        diff_args = []
        for k, v in const_args.items():
            pv = prev_args[k]
            if v != pv: diff_args.append((k, pv, v))
        if diff_args:
            msg = utils.join_head_body(
                f"WARNING: some configuration values are different from resuming checkpoint ({self.resume_ckpt}):",
                '\n'.join([f"{k}: {pv} -> {v}" for k, pv, v in diff_args])
            )
            print(f"-\n{msg}")
    
    def save(self, logger, model, optimizer, scaler):
        filename = f"ckpt-{logger.epoch}.pt"
        ckpt_path = os.path.join(self.path, filename)
        torch.save({
            'logger': logger.state_dict(),
            'model': {
                'type': model.type.value,
                'params': model.state_dict(),
            },
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
        }, ckpt_path)
        print(f"-\nSaved {filename}")
        self.rm_useless_ckpts(self.path, keep_epoch_freq=self.ckpt_keep_freq)
    
    def load_resume(self, logger, model, optimizer, scaler):
        assert self.resume_ckpt
        ckpt = torch.load(os.path.join(self.path, self.resume_ckpt))
        self._compare_args(logger.const_args, ckpt['logger']['args'])
        logger.load_state_dict(ckpt['logger'])
        model.load_state_dict(ckpt['model']['params'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if ckpt['scaler']: scaler.load_state_dict(ckpt['scaler'])
        print(f"-\nSuccessfully loaded checkpoint {self.resume_ckpt}")


class LoadingLogger:

    def __init__(self, wandb):
        self.wandb = wandb
        print("-\nStart loading")
        self.timer = Timer(start=True)
    
    def end(self, args):
        if self.wandb: self.wandb.update_config(args.as_dict())
        print(f"-\n{utils.args_str(args)}")
        print("-\nEnd loading")
        print(f"Loading time: {self.timer.timedelta()}")
    
    @staticmethod
    def datasets(*ds):
        for dataset in ds:
            print(f"-\n{dataset_str(dataset)}")
    
    @staticmethod
    def transforms(*ts):
        for name, transform, in ts:
            print(f"-\n{transform_str(name, transform)}")
    
    @staticmethod
    def dataloaders(*dls):
        for name, dataloader in dls:
            print(f"-\n{utils.dataloader_str(name, dataloader)}")
    
    def model(self, m, log=[]):
        print(f"-\nModel: {m}")
        n_params = utils.count_params(m)
        n_params_str = utils.pretty_number(n_params, 3)
        if self.wandb:
            self.wandb.log_summary("n_parameters", n_params)
            self.wandb.log_summary("n_parameters_str", n_params_str)
        if log: print("-\n" + "\n".join(log))
        print(f"-\nNb parameters: {n_params_str}")
    
    @staticmethod
    def batch_size(bs, ndevices, acc_steps):
        tot_bs = bs * ndevices * acc_steps
        print(f"-\nTotal batch size: {tot_bs} = {bs} (/worker) x {ndevices} (#worker) x {acc_steps} (acc steps)")
    
    @staticmethod
    def optimizer(opt):
        print(f"-\nOptimizer: {opt}")
    
    @staticmethod
    def amp_dtype(dtype):
        if dtype:
            print(f"-\nAMP dtype: {dtype}")
        else:
            print("-\nNot using AMP")


class MetricSeries:
    """
    Track a metric's series of values and compute the average over a window.
    Args:
        - samples_per_step (int, optional):
            Number of samples per step, e.g. batch size.
            The maximum number of steps over which the mean is computed is based on this value.
            We try to gather target_samples points to compute the average.
            If samples_per_step >= target_samples, max_steps would be just 1, but
            if samples_per_step < target_samples, we gather values from more steps
            to get more points and compute a better estimation of the mean, but not
            too many because the distribution changes a bit with each step.
    """

    TARGET_SAMPLES = 1024

    def __init__(self, samples_per_step=None):
        self.samples_per_step = samples_per_step
        if samples_per_step is None:
            max_steps = None
        else:
            max_steps = 1 + max(0, round(math.log2(self.TARGET_SAMPLES / samples_per_step)))
        self.values = deque(maxlen=max_steps)
        self.counts = deque(maxlen=max_steps)
        self.sum_values = 0
        self.sum_counts = 0
    
    def add(self, avg_value, count):
        sum_value = count * avg_value
        self.values.append(sum_value)
        self.counts.append(count)
        self.sum_values = sum(self.values)
        self.sum_counts = sum(self.counts)
    
    def sum_across_procs(self, ddp, device):
        if not ddp.use_ddp: return
        sum_values = torch.tensor([self.sum_values], dtype=torch.float, device=device)
        sum_counts = torch.tensor([self.sum_counts], dtype=torch.int, device=device)
        torch.distributed.all_reduce(sum_values)
        torch.distributed.all_reduce(sum_counts)
        self.sum_values = sum_values.item()
        self.sum_counts = sum_counts.item()
    
    @property
    def mean(self):
        return self.sum_values / self.sum_counts
    
    def state_dict(self):
        return {
            "samples_per_step": self.samples_per_step,
            "values": self.values,
            "counts": self.counts,
            "sum_values": self.sum_values,
            "sum_counts": self.sum_counts,
        }
    
    def load_state_dict(self, state_dict):
        self.samples_per_step = state_dict["samples_per_step"]
        self.values = state_dict["values"]
        self.counts = state_dict["counts"]
        self.sum_values = state_dict["sum_values"]
        self.sum_counts = state_dict["sum_counts"]


class Metrics:

    def __init__(self, use_window):
        self.use_window = use_window
        self.d = OrderedDict()
    
    def clear(self):
        self.d.clear()
    
    def add_val(self, k, v):
        self.d[k] = v
    
    def _new_metric(self, k, count):
        self.d[k] = MetricSeries(count if self.use_window else None)
    
    def add_avg(self, k, v, count):
        if k not in self.d:
            self._new_metric(k, count)
        self.d[k].add(v, count)
    
    def sum_across_procs(self, ddp, device):
        for v in self.d.values():
            if isinstance(v, MetricSeries):
                v.sum_across_procs(ddp, device)
    
    def __len__(self):
        return len(self.d)
    
    def __getitem__(self, key):
        v = self.d[key]
        return v.mean if isinstance(v, MetricSeries) else v
    
    def keys(self):
        return self.d.keys()
    
    def values(self):
        return (self[k] for k in self.keys())
    
    def items(self):
        return zip(self.keys(), self.values())
    
    def state_dict(self):
        res = OrderedDict()
        for k, v in self.d.items():
            res[k] = v.state_dict() if isinstance(v, MetricSeries) else v
        return res
    
    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            if isinstance(v, dict):
                self._new_metric(k, v["samples_per_step"])
                self.d[k].load_state_dict(v)
            else:
                self.d[k] = v


class MetricsLogger:

    def __init__(self, name, use_window):
        self.name = name
        self._head_len = len(self.name)
        self.use_window = use_window
        self.metrics = Metrics(use_window)
        self.units = defaultdict(str)
    
    @property
    def head_len(self):
        return self._head_len if len(self) else 0
    
    def clear(self):
        self.metrics.clear()
    
    def add_units(self, **units):
        self.units.update(units)
    
    def add_val(self, **kwargs):
        for k, v in kwargs.items():
            assert isinstance(v, (float, int))
            self.metrics.add_val(k, v)
    
    def add_avg(self, count, **kwargs):
        for k, v in kwargs.items():
            assert isinstance(v, (float, int))
            self.metrics.add_avg(k, v, count)
    
    def sum_across_procs(self, ddp, device):
        self.metrics.sum_across_procs(ddp, device)
    
    def __len__(self):
        return len(self.metrics)
    
    def __getitem__(self, key):
        return self.metrics[key]
    
    def keys(self):
        return self.metrics.keys()
    
    def values(self):
        return self.metrics.values()
    
    def items(self):
        return self.metrics.items()
    
    def get_str(self, head_fill=0):
        if len(self) == 0: return None
        mtrs = []
        for k, v in self.items():
            unit = self.units[k]
            mtrs.append(f"{k}: {v:.4g}{unit}")
        mtrs = " | ".join(mtrs)
        return f"{self.name: <{head_fill}} - {mtrs}"
    
    def state_dict(self):
        return {
            "metrics": self.metrics.state_dict(),
            "units": self.units,
        }
    
    def load_state_dict(self, state_dict):
        self.metrics.load_state_dict(state_dict["metrics"])
        self.units = state_dict["units"]


class SummaryMetricsLogger:

    def __init__(self, name):
        self.name = name
        self.initialized = False
    
    def track_last(self):
        self.set_objective(None, None)
    
    def set_objective(self, objective_metric, objective):
        self.objective_metric = objective_metric
        self.objective = objective
        self.keep_best = self.objective_metric is not None
        prefix = "best" if self.keep_best else "last"
        comment = ""
        if self.keep_best:
            assert objective in ['maximize', 'minimize']
            self.maximize = objective == 'maximize'
            obj = 'max' if self.maximize else 'min'
            comment = f" ({obj} {objective_metric})"
        head = f"{prefix} {self.name}{comment}"
        self.metrics = MetricsLogger(head, use_window=False)
        self.initialized = True
    
    @property
    def head_len(self):
        return self.metrics.head_len
    
    def add_units(self, **units):
        self.metrics.add_units(**units)

    def update(self, epoch, metrics):
        assert self.initialized
        if self.objective_metric in self.metrics.keys() and self.keep_best:
            new_v = metrics[self.objective_metric]
            v = self.metrics[self.objective_metric]
            if self.maximize and not (new_v > v): return
            elif not self.maximize and not (new_v < v): return
        self.metrics.add_val(epoch=epoch, **OrderedDict(metrics.items()))
    
    def get_str(self, head_fill=0):
        assert self.initialized
        return self.metrics.get_str(head_fill)
    
    def state_dict(self):
        assert self.initialized
        return {
            "objective_metric": self.objective_metric,
            "objective": self.objective,
            "metrics": self.metrics.state_dict(),
        }
    
    def load_state_dict(self, state_dict):
        if not self.initialized:
            self.set_objective(state_dict["objective_metric"], state_dict["objective"])
        self.metrics.load_state_dict(state_dict["metrics"])


class IterLogger:

    def __init__(
        self,
        name,
        iter_name,
        iter_epochs,
        steps_per_iter,
        ckpt_freq,
        print_freq,
        glob,
        run,
        ddp,
        device,
        use_window=True,
        objective_metric=None,
        objective=None,
    ):
        self.name = name
        self.use_window = use_window
        assert utils.all_none_or_not(objective_metric, objective)
        self.track_best = objective_metric is not None
        self.ckpt_freq = ckpt_freq
        self.print_freq = print_freq
        self.glob = glob
        self.run = run
        self.ddp = ddp
        self.device = device

        n_iters = sum(iter_epochs)
        self.tp = TimeProgress(n_iters, steps_per_iter, iter_name)
        self.glob.tp.register(self.tp, iter_epochs)

        if self.run.wandb:
            if self.use_window:
                # This is the minimum positive int f s.t. n_logs = 1 + ceil(n_steps / f) <= max_n_logs.
                self.wandb_log_step_freq = math.ceil(self.tp.n_steps / (self.run.wandb.MAX_N_LOGS - 1))
                step_freq_str = utils.quantity_str(self.wandb_log_step_freq, "step", "steps")
                iter_freq_str = utils.quantity_str(self.wandb_log_step_freq / self.tp.steps_per_iter, "iteration", "iterations", fmt=".4g")
                print(f"-\n{self.name} logger logs in WandB every {step_freq_str} ({iter_freq_str}).")
            else:
                self.wandb_log_iter_freq = math.ceil(self.tp.n_iters / (self.run.wandb.MAX_N_LOGS - 1))
                if self.wandb_log_iter_freq:
                    iter_freq_str = utils.quantity_str(self.wandb_log_iter_freq, "iteration", "iterations")
                    print(f"-\n{self.name} logger logs in WandB every {iter_freq_str}.")

        self.metrics = MetricsLogger(self.name, self.use_window)
        if self.track_best:
            assert not self.use_window
            self.last_metrics = SummaryMetricsLogger(self.name)
            self.best_metrics = SummaryMetricsLogger(self.name)
            self.last_metrics.track_last()
            self.best_metrics.set_objective(objective_metric, objective)
            self.glob.register_last_metrics(self.last_metrics)
            self.glob.register_best_metrics(self.best_metrics)
        else:
            self.last_metrics = self.best_metrics = None
        
        self.last_saved_epoch = None
    
    @property
    def next_save_epoch(self):
        if not self.ckpt_freq: return None
        last_saved_epoch = self.last_saved_epoch if self.last_saved_epoch else 0
        if last_saved_epoch == self.glob.n_epochs: return None
        else: return min(last_saved_epoch + self.ckpt_freq, self.glob.n_epochs)
    
    def _resume(self):
        self.tp.resume()
    
    def _pause(self):
        self.tp.pause()
    
    def add_units(self, **units):
        self.metrics.add_units(**units)
        if self.track_best:
            self.last_metrics.add_units(**units)
            self.best_metrics.add_units(**units)
    
    def sum_metrics_across_procs(self):
        if not self.ddp.use_ddp: return
        if not self.use_window and self.tp.local_step == self.tp.steps_per_iter - 1:
            self.metrics.sum_across_procs(self.ddp, self.device)
    
    def update_summary_metrics(self):
        if self.track_best and self.tp.local_step == self.tp.steps_per_iter - 1:
            self.last_metrics.update(int(self.glob.epoch), self.metrics)
            self.best_metrics.update(int(self.glob.epoch), self.metrics)
    
    def wandb_log(self):
        if self.run.wandb is None: return
        if self.use_window:
            # Log every k steps and at last step.
            if self.tp.step % self.wandb_log_step_freq == 0 or self.tp.step == self.tp.n_steps - 1:
                self.run.wandb.log(
                    self.glob.epoch,
                    self.metrics,
                    time_left=self.glob.remaining_time,
                    time_left_str=self.glob.pretty_remaining_time,
                )
        else:
            # Log every k iterations and at last iteration.
            if self.tp.local_step == self.tp.steps_per_iter - 1:
                if self.tp.iter % self.wandb_log_iter_freq == 0 or self.tp.iter == self.tp.n_iters - 1:
                    if self.track_best:
                        best_metric_name = self.best_metrics.objective_metric
                        best_metric = self.best_metrics.metrics[best_metric_name]
                    else:
                        best_metric_name = best_metric = None
                    self.run.wandb.log(
                        self.glob.epoch,
                        self.metrics,
                        best_metric_name,
                        best_metric,
                        time_left=self.glob.remaining_time,
                        time_left_str=self.glob.pretty_remaining_time,
                    )
    
    def add_val(self, **kwargs):
        self.metrics.add_val(**kwargs)
        self.update_summary_metrics()
        self.wandb_log()
    
    def add_avg(self, count, **kwargs):
        self.metrics.add_avg(count, **kwargs)
        self.sum_metrics_across_procs()
        self.update_summary_metrics()
        self.wandb_log()
    
    def _ckpt_msg(self):
        last_epoch = self.glob.last_saved_epoch
        last_ckpt = f"epoch {last_epoch}" if last_epoch is not None else "none"
        last_ckpt = f"last ckpt: {last_ckpt}"

        next_epoch = self.glob.next_save_epoch
        if last_epoch is None and next_epoch is None: return "no ckpt"
        if next_epoch is not None:
            next_time, next_date = self.glob.tp.minimal_pretty_time_progress(next_epoch)
            next_ckpt = f"epoch {next_epoch} | {next_date} (in {next_time})"
        else:
            next_ckpt = "none"
        next_ckpt = f"next ckpt: {next_ckpt}"

        return "\n".join([last_ckpt, next_ckpt])
    
    def _log_msg(self):

        # Global time progress
        gtp = self.glob.tp.get_str()
        ckpt_tp = self._ckpt_msg()
        gtp = "\n".join([gtp, ckpt_tp])
        gtp = utils.join_head_body("GLOBAL", gtp)

        # Local time progress
        tp = self.tp.get_str(local=True)
        tp = utils.join_head_body(
            f"{self.name.upper()} ITERATION - {self.tp.iter + 1}/{self.tp.n_iters}", tp
        )

        # Metrics
        head_fill = max(
            getattr(m, 'head_len') for m in [self.metrics, *self.glob.last_metrics, *self.glob.best_metrics]
        )
        mtr = []
        for last_metrics, best_metrics in zip(self.glob.last_metrics, self.glob.best_metrics):
            mtr.append(best_metrics.get_str(head_fill))
            mtr.append(last_metrics.get_str(head_fill))
        mtr.append(self.metrics.get_str(head_fill))
        mtr = [m for m in mtr if m] # remove empty rows
        mtr = "\n".join(mtr)
        mtr = utils.join_head_body("METRICS", mtr)

        # Join
        msg = "\n-\n".join([gtp, tp, mtr])
        sep = '-' * utils.max_line(msg)
        return "\n".join([sep, msg])
    
    def _save_ckpt(self):
        self.last_saved_epoch = int(self.glob.epoch)
        self.run.save(self.glob.logger, self.glob.model, self.glob.optimizer, self.glob.scaler)
    
    @property
    def step(self):
        return self.tp.local_step
    
    @property
    def steps_per_iter(self):
        return self.tp.steps_per_iter
    
    def log_iter(self, iterable):
        # Clear metrics at each iteration when there is no window for MetricSeries.
        if not self.use_window: self.metrics.clear()
        self.tp.start_iter()
        for element in iterable:
            yield element
            self.tp.end_step()
            if self.step % self.print_freq == 0 or self.step == self.steps_per_iter:
                print(self._log_msg())
            # Do not rely on the iterable's end to stop,
            # as self.steps_per_iter < len(iterable) is possible.
            if self.step == self.steps_per_iter: break
        self.tp.end_iter()
        epoch = int(self.glob.epoch)
        if epoch == self.next_save_epoch or epoch == self.glob.last_saved_epoch: # save eval results too
            self._save_ckpt()
    
    def state_dict(self):
        return {
            "tp": self.tp.state_dict(),
            "metrics": self.metrics.state_dict(),
            "last_metrics": self.last_metrics.state_dict() if self.last_metrics else None,
            "best_metrics": self.best_metrics.state_dict() if self.best_metrics else None,
            "last_saved_epoch": self.last_saved_epoch,
        }
    
    def load_state_dict(self, state_dict):
        self.tp.load_state_dict(state_dict["tp"])
        self.metrics.load_state_dict(state_dict["metrics"])
        if self.last_metrics: self.last_metrics.load_state_dict(state_dict["last_metrics"])
        if self.best_metrics: self.best_metrics.load_state_dict(state_dict["best_metrics"])
        self.last_saved_epoch = state_dict["last_saved_epoch"]


class GlobalIterLogger:

    def __init__(self, logger, model, optimizer, scaler):
        self.logger = logger
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.tp = GlobalTimeProgress()
        self.last_metrics = []
        self.best_metrics = []
    
    @property
    def epoch(self):
        return self.tp.epoch
    
    @property
    def n_epochs(self):
        return self.tp.n_epochs
    
    @property
    def remaining_time(self):
        return self.tp.time_progress()[1]
    
    @property
    def pretty_remaining_time(self):
        return self.tp.minimal_pretty_time_progress()[0]
    
    @property
    def last_saved_epoch(self):
        return self.logger.train.last_saved_epoch
    
    @property
    def next_save_epoch(self):
        return self.logger.train.next_save_epoch
    
    def register_last_metrics(self, last_metrics):
        self.last_metrics.append(last_metrics)
    
    def register_best_metrics(self, best_metrics):
        self.best_metrics.append(best_metrics)


class Logger:
    """Logger that handles resuming, WandB logging and progress printing."""

    def __init__(
        self,
        job_id,
        ddp,
        device,
        run_name,
        resume_ckpt,
        use_wandb,
        n_epochs,
        ckpt_freq,
        ckpt_keep_freq,
        eval_freq,
        eval_steps_per_iter,
        print_freq_train,
        print_freq_eval,
        eval_objective,
    ):
        self.job_id = job_id
        self.ddp = ddp
        self.device = device
        if not ddp.is_master:
            ckpt_freq = None
            use_wandb = False
        self.run = Run(
            run_name,
            resume_ckpt,
            save_ckpt=ckpt_freq is not None,
            ckpt_keep_freq=ckpt_keep_freq,
            use_wandb=use_wandb,
        )
        self.loading = LoadingLogger(self.run.wandb)
        self.n_epochs = n_epochs
        self.ckpt_freq = ckpt_freq
        self.eval_epochs = self._get_eval_epochs(n_epochs, eval_freq)
        self.eval_steps_per_iter = eval_steps_per_iter
        self.print_freq_train = print_freq_train
        self.print_freq_eval = print_freq_eval
        self.eval_objective = eval_objective
    
    @staticmethod
    def _get_train_epochs(n_epochs):
        res = [True] * (n_epochs + 1)
        res[0] = False
        return res

    @staticmethod
    def _get_eval_epochs(n_epochs, eval_freq):
        res = [False] * (n_epochs + 1)
        if eval_freq is None: return res
        res[0] = True
        for i in range(1, n_epochs + 1):
            if i % eval_freq == 0: res[i] = True
        res[-1] = True
        return res
    
    def init_iter_loggers(self):
        self.train = IterLogger(
            "training",
            TimeProgress.EPOCHS_STR,
            self._get_train_epochs(self.n_epochs),
            self.train_steps_per_iter,
            self.ckpt_freq,
            self.print_freq_train,
            self.glob,
            self.run,
            self.ddp,
            self.device,
        )
        if self.eval_steps_per_iter is not None:
            eval_steps_per_iter = utils.clamp(self.eval_steps_per_iter, 1, self.n_batches_eval)
        else:
            eval_steps_per_iter = self.n_batches_eval
        objective_metric, objective = self.eval_objective
        self.eval = IterLogger(
            "validation",
            "evals",
            self.eval_epochs,
            eval_steps_per_iter,
            None,
            self.print_freq_eval,
            self.glob,
            self.run,
            self.ddp,
            self.device,
            use_window=False,
            objective_metric=objective_metric,
            objective=objective,
        )
    
    def end_loading(self, vargs, cargs, n_batches_train, grad_acc_steps, n_batches_eval, model, optimizer, scaler, amp_dtype):
        self.const_args = cargs
        self.args = utils.merge_args(vargs, cargs)
        self.train_steps_per_iter = n_batches_train - n_batches_train % grad_acc_steps
        self.grad_acc_steps = grad_acc_steps
        self.n_batches_eval = n_batches_eval
        self.amp_dtype = amp_dtype
        self.glob = GlobalIterLogger(self, model, optimizer, scaler)
        if self.run.resume_ckpt:
            self.run.load_resume(self, model, optimizer, scaler)
        else:
            self.init_iter_loggers()
        self.loading.end(self.args)
    
    @property
    def fractional_epoch(self):
        return self.glob.epoch
    
    @property
    def epoch(self):
        return int(self.glob.epoch)
    
    def start_train(self):
        if self.run.resume_ckpt:
            print(f'-\nResume training from epoch {self.epoch}')
        else:
            print(f'-\nStart new training')
        self.train._resume()
    
    def end_train(self):
        self.train._pause()
        print("-\nEnd training")
        print(
            f"Training time: {self.glob.tp.timedelta()}"
            f" ({self.train.tp.timedelta()} pure training + {self.eval.tp.timedelta()} eval)"
        )
        if self.run.wandb: self.run.wandb.finish()
    
    def resume_eval(self):
        self.train._pause()
        self.eval._resume()
    
    def pause_eval(self):
        self.eval._pause()
        self.train._resume()
    
    def state_dict(self):
        date = Date.to_tuple(Date.now())
        date_str = Date.to_str(Date.now())

        # Save some useful info into wandb.
        if self.run.wandb:
            self.run.wandb.log_summary("job_id", self.job_id)
            self.run.wandb.log_summary("date", date)
            self.run.wandb.log_summary("date_str", date_str)
            self.run.wandb.log_summary("total_time", self.glob.tp.time())
            self.run.wandb.log_summary("total_time_str", str(self.glob.tp.timedelta()))
            self.run.wandb.log_summary("train_time", self.train.tp.time())
            self.run.wandb.log_summary("train_time_str", str(self.train.tp.timedelta()))
            self.run.wandb.log_summary("eval_time", self.eval.tp.time())
            self.run.wandb.log_summary("eval_time_str", str(self.eval.tp.timedelta()))
            self.run.wandb.log_summary("world_size", self.ddp.world_size)
            self.run.wandb.log_summary("amp_dtype", str(self.amp_dtype))
            self.run.wandb.log_summary("grad_acc_steps", self.grad_acc_steps)

        epoch = self.epoch if self.epoch != self.eval.last_saved_epoch else self.epoch + 1
        return {
            "job_id": self.job_id,
            "date": date,
            "world_size": self.ddp.world_size,
            "amp_dtype": str(self.amp_dtype),
            "grad_acc_steps": self.grad_acc_steps,
            "args": self.args,
            "eval_epochs": self.eval_epochs[:epoch],
            "wandb": self.run.wandb.state_dict() if self.run.wandb else None,
            "train": self.train.state_dict(),
            "eval": self.eval.state_dict(),
        }
    
    def load_state_dict(self, state_dict):
        eval_epochs = state_dict["eval_epochs"]
        self.eval_epochs[:len(eval_epochs)] = eval_epochs
        self.init_iter_loggers()

        if self.run.wandb: self.run.wandb.load_state_dict(state_dict["wandb"])
        self.train.load_state_dict(state_dict["train"])
        self.eval.load_state_dict(state_dict["eval"])