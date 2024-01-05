import torch
import torch.backends.cuda
import torch.backends.cudnn
import torch.utils.data
import argparse
import utils
from datasets import create_dataset, create_transform
from models import create_ssl, create_knn
from optimizers import create_optimizer
from engine_ssl import train_one_epoch, evaluate
from logger import Logger


def get_args_parser():
    parser = argparse.ArgumentParser("SSL training", add_help=False)
    parser.add_argument("--job_id", default=None, type=int)
    parser.add_argument(
        "--var_config",
        default="configs/main_train_ssl/variable/c1.yaml",
        type=str,
    )
    parser.add_argument(
        "--const_config",
        default="configs/main_train_ssl/constant/c1.yaml",
        type=str,
    )
    return parser


def get_variable_args_parser():
    parser = utils.ArgumentParser("SSL training - Variable parameters", add_help=False)
    
    # If resume_ckpt is null, will resume from last ckpt.
    # Not giving the option to start from scratch forces the user to delete the ckpts by hand (safer).
    parser.add_argument("--run_name", type=str) # optional
    parser.add_argument("--resume_ckpt", type=str) # optional

    parser.add_argument("--seed", type=int)

    parser.add_argument("--device", choices=['cpu', 'cuda'], type=str)
    parser.add_argument("--use_amp", type=bool)
    parser.add_argument("--n_workers", type=int)

    parser.add_argument("--ckpt.freq", type=int) # optional
    parser.add_argument("--ckpt.keep_freq", type=int) # optional
    parser.add_argument("--eval_freq", type=int) # optional
    parser.add_argument("--print_freq.train", type=int)
    parser.add_argument("--print_freq.eval", type=int)

    return parser


def get_constant_args_parser():
    parser = utils.ArgumentParser("SSL training - Constant parameters", add_help=False)

    parser.add_argument("--use_wandb", type=bool)

    # Dataset
    parser.add_argument("--dataset.config_dir", type=str)
    parser.add_argument("--dataset.train.config", type=str)
    parser.add_argument("--dataset.eval.config", type=str)
    parser.add_argument("--dataset.transform.train.config", type=str)
    parser.add_argument("--dataset.transform.eval.config", type=str)

    # Model
    parser.add_argument("--model.config_dir", type=str)
    parser.add_argument("--model.config", type=str)

    # k-NN
    parser.add_argument("--knn.config", type=str)
    parser.add_argument("--knn.batch_size", type=int)

    # Optimizer
    parser.add_argument("--optimizer.config_dir", type=str)
    parser.add_argument("--optimizer.config", type=str)
    parser.add_argument("--optimizer.lr.scale_with_batch", type=bool)
    parser.add_argument("--optimizer.lr.lr", type=float)
    parser.add_argument("--optimizer.lr.decay.use", type=bool)
    parser.add_argument("--optimizer.lr.decay.warmup_epochs", type=int)
    parser.add_argument("--optimizer.lr.decay.epochs", type=int) # optional, if null, set to n_epochs
    parser.add_argument("--optimizer.lr.decay.min", type=float)
    parser.add_argument("--optimizer.lr.decay.layer_decay", type=float) # optional
    parser.add_argument("--optimizer.weight_decay", type=float)

    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--batch_size.train.total", type=int)
    parser.add_argument("--batch_size.train.per_worker", type=int)
    parser.add_argument("--batch_size.eval.per_worker", type=int)

    return parser


def main(job_id, vargs, cargs):
    ddp = utils.DDP()
    
    args = utils.merge_args(vargs, cargs)
    device = torch.device({'cpu': 'cpu', 'cuda': ddp.local_rank}[args.device])
    utils.fix_seed(args.seed + ddp.rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    logger = Logger(
        job_id,
        ddp,
        device,
        args.run_name,
        args.resume_ckpt,
        args.use_wandb,
        args.n_epochs,
        args.ckpt.freq,
        args.ckpt.keep_freq,
        args.eval_freq,
        None,
        args.print_freq.train,
        args.print_freq.eval,
        eval_objective=('knn_acc1', 'maximize'),
    )

    # Dataset
    trainset = create_dataset(args.dataset.config_dir, args.dataset.train.config, train=True)
    trainset_knn = create_dataset(args.dataset.config_dir, args.dataset.eval.config, train=True)
    valset = create_dataset(args.dataset.config_dir, args.dataset.eval.config, train=False)
    cargs.dataset.train.update(trainset.args)
    cargs.dataset.eval.update(valset.args)
    logger.loading.datasets(trainset, trainset_knn, valset)

    # Data transform
    train_transform = create_transform(trainset, args.dataset.transform.train.config)
    val_transform = create_transform(valset, args.dataset.transform.eval.config)
    cargs.dataset.transform.train.update(train_transform.args)
    cargs.dataset.transform.eval.update(val_transform.args)
    logger.loading.transforms(
        ("Train Transform", train_transform),
        ("Validation Transform", val_transform),
    )

    trainset.transform = train_transform
    trainset_knn.transform = val_transform
    valset.transform = val_transform

    # Data sampler
    if ddp.use_ddp:
        train_sampler = torch.utils.data.DistributedSampler(
            trainset, num_replicas=ddp.world_size, rank=ddp.rank, shuffle=True,
        )
        train_knn_sampler = utils.DistributedSamplerNoDuplicate(
            trainset_knn, num_replicas=ddp.world_size, rank=ddp.rank, shuffle=False,
        )
        val_sampler = utils.DistributedSamplerNoDuplicate(
            valset, num_replicas=ddp.world_size, rank=ddp.rank, shuffle=False,
        )
    else:
        train_sampler = torch.utils.data.RandomSampler(trainset)
        train_knn_sampler = torch.utils.data.SequentialSampler(trainset_knn)
        val_sampler = torch.utils.data.SequentialSampler(valset)

    # Data loader
    trainloader = torch.utils.data.DataLoader(
        trainset,
        sampler=train_sampler,
        batch_size=args.batch_size.train.per_worker,
        num_workers=args.n_workers,
        pin_memory=True,
        drop_last=True,
    )
    train_knn_loader = torch.utils.data.DataLoader(
        trainset_knn,
        sampler=train_knn_sampler,
        batch_size=args.batch_size.eval.per_worker,
        num_workers=args.n_workers,
        pin_memory=True,
        drop_last=False,
    )
    valloader = torch.utils.data.DataLoader(
        valset,
        sampler=val_sampler,
        batch_size=args.batch_size.eval.per_worker,
        num_workers=args.n_workers,
        pin_memory=True,
        drop_last=False,
    )
    logger.loading.dataloaders(
        ("Train DataLoader", trainloader),
        ("Validation DataLoader", valloader),
    )

    # Model
    model = create_ssl(
        args.model.config_dir,
        args.model.config,
        train_transform.channels,
        train_transform.img_size,
    )
    cargs.model.update(model.args)
    model.to(device)
    logger.loading.model(model)

    # knn classifier
    encoder = model.create_encoder()
    encoder.to(device)
    knn = create_knn(args.knn.config, trainset.n_classes, args.knn.batch_size)

    # Batch size
    tot_batch_size = args.batch_size.train.total
    batch_size = args.batch_size.train.per_worker
    assert tot_batch_size % (batch_size * ddp.world_size) == 0
    grad_acc_steps = tot_batch_size // (batch_size * ddp.world_size)
    logger.loading.batch_size(batch_size, ddp.world_size, grad_acc_steps)

    # Learning rate
    lr = args.optimizer.lr.lr
    if args.optimizer.lr.scale_with_batch:
        lr = lr * tot_batch_size / 256
    use_decay = args.optimizer.lr.decay.use
    decay_epochs = args.optimizer.lr.decay.epochs
    if decay_epochs is None: decay_epochs = args.n_epochs
    set_lr = utils.LearningRateSetter(
        lr,
        use_decay,
        args.optimizer.lr.decay.warmup_epochs,
        decay_epochs,
        args.optimizer.lr.decay.min,
    )

    # Optimizer
    layer_decay = args.optimizer.lr.decay.layer_decay if use_decay else None
    optimizer = create_optimizer(
        args.optimizer.config_dir,
        args.optimizer.config,
        model,
        lr,
        layer_decay,
        args.optimizer.weight_decay,
    )
    logger.loading.optimizer(optimizer)

    # AMP
    amp_dtype = utils.get_amp_dtype(args.use_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype == torch.float16)
    logger.loading.amp_dtype(amp_dtype)

    # DDP
    if ddp.use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
        encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[device])
        raw_model = model.module
    else:
        raw_model = model

    logger.end_loading(
        vargs=vargs,
        cargs=cargs,
        n_batches_train=len(trainloader),
        grad_acc_steps=grad_acc_steps,
        n_batches_eval=len(valloader),
        model=raw_model,
        optimizer=optimizer,
        scaler=scaler,
        amp_dtype=amp_dtype,
    )

    # Train
    logger.start_train()
    if logger.eval_epochs[logger.epoch] and logger.eval.last_saved_epoch != logger.epoch:
        evaluate(train_knn_loader, valloader, device, amp_dtype, raw_model, encoder, knn, logger)
    for _ in range(logger.epoch, args.n_epochs):
        if ddp.use_ddp: trainloader.sampler.set_epoch(logger.epoch)
        train_one_epoch(trainloader, device, amp_dtype, model, grad_acc_steps, optimizer, set_lr, scaler, logger)
        if logger.eval_epochs[logger.epoch]:
            evaluate(train_knn_loader, valloader, device, amp_dtype, raw_model, encoder, knn, logger)
    logger.end_train()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    vparser = get_variable_args_parser()
    cparser = get_constant_args_parser()
    vargs = vparser.parse_config(args.var_config)
    cargs = cparser.parse_config(args.const_config)
    main(args.job_id, vargs, cargs)