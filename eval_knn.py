import torch
import torch.backends.cuda
import torch.backends.cudnn
import torch.utils.data
import utils
from logger import Run
from datasets import create_dataset, create_transform
from models import create_ssl, create_knn
from tqdm import tqdm
import os


############################################################

epochs = [0, 100, 200, 300, 400, 500, 600, 700, 800]

use_amp = True
device_name = 'cuda'
n_workers = 10
use_amp = True
seed = 0

dataset_config_dir = "imagenet"
dataset_config = "full.yaml"
transform_config = "eval.yaml"

model_config_dir = "mae_vit"
model_config = "imagenet/base-16.yaml"
model_run = "ssl/in1k/mae/vit/exp1/run1"

knn_config = "k=20.yaml"
knn_batch_size = 512

batch_size_per_worker = 512

############################################################


def main():

    ddp = utils.DDP()

    device = torch.device({'cpu': 'cpu', 'cuda': ddp.local_rank}[device_name])
    utils.fix_seed(seed + ddp.rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Dataset
    trainset = create_dataset(dataset_config_dir, dataset_config, train=True)
    valset = create_dataset(dataset_config_dir, dataset_config, train=False)

    # Data transform
    transform = create_transform(valset, transform_config)
    trainset.transform = transform
    valset.transform = transform

    # Data sampler
    if ddp.use_ddp:
        train_sampler = utils.DistributedSamplerNoDuplicate(
            trainset, num_replicas=ddp.world_size, rank=ddp.rank, shuffle=False,
        )
        val_sampler = utils.DistributedSamplerNoDuplicate(
            valset, num_replicas=ddp.world_size, rank=ddp.rank, shuffle=False,
        )
    else:
        train_sampler = torch.utils.data.SequentialSampler(trainset)
        val_sampler = torch.utils.data.SequentialSampler(valset)

    # Data loader
    trainloader = torch.utils.data.DataLoader(
        trainset,
        sampler=train_sampler,
        batch_size=batch_size_per_worker,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=False,
    )
    valloader = torch.utils.data.DataLoader(
        valset,
        sampler=val_sampler,
        batch_size=batch_size_per_worker,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Model
    model = create_ssl(
        model_config_dir,
        model_config,
        trainloader.dataset.transform.channels,
        trainloader.dataset.transform.img_size,
    )

    # knn classifier
    encoder = model.create_encoder()
    encoder.to(device)
    knn = create_knn(knn_config, trainloader.dataset.n_classes, knn_batch_size)

    # AMP
    amp_dtype = utils.get_amp_dtype(use_amp)

    # DDP
    if ddp.use_ddp:
        encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[device])

    # Evaluate
    res = []
    for i, epoch in enumerate(epochs):
        print(f"-\n{i+1}/{len(epochs)}")
        with torch.no_grad():
            
            # Load params from ckpt.
            if epoch != 0:
                ckpt = os.path.join(model_run, Run.ckpt_name(epoch))
                ckpt, _ = Run.load_ckpt(ckpt)
                model.load_state_dict(ckpt['model']['params'])
                raw_encoder = encoder.module if ddp.use_ddp else encoder
                model.load_encoder(raw_encoder)
            
            encoder.eval()

            # Extract train features.
            print("Extracting training features...")
            for samples, targets in tqdm(trainloader):
                samples = samples.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.autocast(device.type, amp_dtype, enabled=amp_dtype is not None):
                    features = encoder(samples)
                knn.add_train_features(features, targets)
            knn.gather_train_features(ddp)

            # Extract val features.
            print("Extracting test features...")
            for samples, targets in tqdm(valloader):
                samples = samples.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.autocast(device.type, amp_dtype, enabled=amp_dtype is not None):
                    features = encoder(samples)
                knn.add_eval_features(features, targets)

            # Classify
            knn.gather_eval_features(ddp)
            acc1, acc5 = knn.classify()
            res.append((acc1, acc5))
            print(f"epoch: {epoch} | acc1: {acc1:.2f}% | acc5: {acc5:.2f}%")
            knn.reset()
    return res


if __name__ == "__main__":
    res = main()
    print("-")
    for epoch, (acc1, acc5) in zip(epochs, res):
        print(f"epoch: {epoch} | acc1: {acc1:.2f}% | acc5: {acc5:.2f}%")