import torch
import math
from logger import Timer


def train_one_epoch(dataloader, device, amp_dtype, model, grad_acc_steps, optimizer, set_lr, scaler, logger):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for step, (samples, _) in enumerate(logger.train.log_iter(dataloader)):
        take_step = (step + 1) % grad_acc_steps == 0
        if logger.ddp.use_ddp:
            model.require_backward_grad_sync = take_step
        
        samples = samples.to(device, non_blocking=True)

        with torch.autocast(device.type, amp_dtype, enabled=amp_dtype is not None):
            loss, loss_name = model(samples)[:2]
            loss /= grad_acc_steps
        
        scaler.scale(loss).backward()
        if take_step:
            lr = set_lr(optimizer, logger.fractional_epoch)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            logger.train.add_val(lr=lr)

        loss_val = loss.item() * grad_acc_steps
        assert math.isfinite(loss_val)
        logger.train.add_avg(len(samples), **{loss_name: loss_val})


@torch.no_grad()
def evaluate(trainloader, valloader, device, amp_dtype, raw_model, encoder, knn, logger):
    logger.resume_eval()
    logger.eval.add_units(knn_acc1="%", knn_acc5="%")

    raw_encoder = encoder.module if logger.ddp.use_ddp else encoder
    raw_model.load_encoder(raw_encoder)
    encoder.eval()

    # Extract train features.
    print("-\nExtracting training features for k-NN...")
    timer = Timer(start=True)
    for samples, targets in trainloader:
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device.type, amp_dtype, enabled=amp_dtype is not None):
            features = encoder(samples)
        knn.add_train_features(features, targets)
    knn.gather_train_features(logger.ddp)
    print(f"Extraction time: {timer.timedelta()}")

    # Extract val features.
    for samples, targets in logger.eval.log_iter(valloader):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device.type, amp_dtype, enabled=amp_dtype is not None):
            features = encoder(samples)
        knn.add_eval_features(features, targets)

        # Classify
        if logger.eval.step == logger.eval.steps_per_iter - 1:
            knn.gather_eval_features(logger.ddp)
            knn_acc1, knn_acc5 = knn.classify()
            logger.eval.add_val(knn_acc1=knn_acc1, knn_acc5=knn_acc5)
            knn.reset()

    logger.pause_eval()