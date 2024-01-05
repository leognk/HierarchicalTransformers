import torch
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


def create_mixup(mixup, cutmix, smoothing, n_classes):
    """Return mixup function and criterion."""
    if mixup or cutmix:
        return (
            Mixup(
                mixup_alpha=mixup,
                cutmix_alpha=cutmix,
                label_smoothing=smoothing,
                num_classes=n_classes,
            ),
            SoftTargetCrossEntropy(),
        )
    elif smoothing:
        return None, LabelSmoothingCrossEntropy(smoothing)
    return None, torch.nn.CrossEntropyLoss()