import torch
import utils
import math


def train_one_epoch(dataloader, device, amp_dtype, model, grad_acc_steps, optimizer, set_lr, scaler, mixup, criterion, logger):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for step, (samples, targets) in enumerate(logger.train.log_iter(dataloader)):
        take_step = (step + 1) % grad_acc_steps == 0
        if logger.ddp.use_ddp:
            model.require_backward_grad_sync = take_step

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup is not None:
            samples, targets = mixup(samples, targets)

        with torch.autocast(device.type, amp_dtype, enabled=amp_dtype is not None):
            preds = model(samples)
            # preds might contain auxiliary predictions in addition to the final prediction.
            n_aux = preds.shape[0] // targets.shape[0]
            loss = criterion(preds, utils.repeat_tensor(targets, n_aux))
            loss /= grad_acc_steps
            preds = preds[:targets.shape[0]]

        scaler.scale(loss).backward()
        if take_step:
            lr = set_lr(optimizer, logger.fractional_epoch)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            logger.train.add_val(lr=lr)

        loss_val = loss.item() * grad_acc_steps
        assert math.isfinite(loss_val)
        logger.train.add_avg(len(samples), loss=loss_val)


@torch.no_grad()
def evaluate(dataloader, device, amp_dtype, model, logger):
    logger.resume_eval()
    logger.eval.add_units(acc1="%", acc5="%")

    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    for samples, targets in logger.eval.log_iter(dataloader):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device.type, amp_dtype, enabled=amp_dtype is not None):
            preds = model(samples)
            # preds might contain auxiliary predictions in addition to the final prediction.
            n_aux = preds.shape[0] // targets.shape[0]
            loss = criterion(preds, utils.repeat_tensor(targets, n_aux))
            preds = preds[:targets.shape[0]]

        acc1, acc5 = utils.accuracy(preds, targets, topk=(1, 5))
        logger.eval.add_avg(len(samples), loss=loss.item(), acc1=acc1, acc5=acc5)

    logger.pause_eval()