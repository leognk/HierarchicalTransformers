import torch
from PIL.Image import Image
import numpy as np
import matplotlib.pyplot as plt


def cumsum(lst):
    res = [lst[0]]
    for i in range(1, len(lst)):
        res.append(res[-1] + lst[i])
    return res


def cumdiff(lst):
    res = [lst[0]]
    for i in range(1, len(lst)):
        res.append(lst[i] - lst[i - 1])
    return res


def to_pair(x):
    return (x, x) if x is not None else None


def get_img_size(x):
    """
    Args:
        - x (PIL Image, NumPy ndarray or PyTorch Tensor):
            - if ndarray: shape=[h w c]
            - if Tensor: shape=[c h w]
    Returns: [h w]
    """
    if isinstance(x, Image):
        return x.height, x.width
    elif isinstance(x, np.ndarray):
        return x.shape[:2]
    elif isinstance(x, torch.Tensor):
        return x.shape[-2:]
    else:
        raise TypeError(f"Unsupported type: {type(x)}")


def torch_to_np_img(x, mean=None, std=None):
    """
    Args:
        - x (Tensor): shape: [c h w]
        - mean (sequence[float])
        - std (sequence[float])
    """
    x = x.detach().cpu().numpy().transpose((1, 2, 0))
    if mean is not None and std is not None:
        if not isinstance(mean, np.ndarray): mean = np.array(mean)
        if not isinstance(std, np.ndarray): std = np.array(std)
        x = x * std + mean
    x = np.clip(x, 0, 1)
    return x


def plot_img(img, mean=None, std=None, title="", figsize=None):
    """
    Args:
        - img (PIL Image, NumPy ndarray or PyTorch Tensor):
            - if ndarray: shape=[h w c]
            - if Tensor: shape=[c h w]
        - mean (sequence[float]): only used for tensor img
        - std (sequence[float]): only used for tensor img
        - title (str, optional)
        - figsize (int, optional)
    """
    if isinstance(img, torch.Tensor):
        img = torch_to_np_img(img, mean, std)
    elif not isinstance(img, np.ndarray):
        img = np.array(img)
    _, ax = plt.subplots(figsize=to_pair(figsize)) # maintains aspect ratio
    ax.set_axis_off()
    ax.set_title(title)
    ax.imshow(img)
    plt.show()