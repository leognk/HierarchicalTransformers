import torchvision.transforms as T
import torchvision.transforms.functional as F
from timm.data.auto_augment import rand_augment_transform, augment_and_mix_transform, auto_augment_transform
from timm.data.transforms import str_to_pil_interp
import utils


class ResizeKeepRatio:
    """
    Args:
        - scale_size (sequence[int]):
            Warning: the output size won't necessarily be matched to this size.
        - interpolation (InterpolationMode)
    """

    def __init__(self, scale_size, interpolation):
        self.scale_size = scale_size
        self.interpolation = interpolation

    def get_out_size(self, img):
        in_h, in_w = img.size[::-1]
        h, w = self.scale_size
        ratio_h = in_h / h
        ratio_w = in_w / w
        ratio = min(ratio_h, ratio_w)
        out_size = (round(in_h / ratio), round(in_w / ratio))
        return out_size

    def __call__(self, img):
        out_size = self.get_out_size(img)
        img = F.resize(img, out_size, self.interpolation)
        return img

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size}, interpolation={self.interpolation.value})"


class ResizeCenterCrop:
    """
    Resize while preserving the aspect ratio and center crop to the target size
    the excess pixels from the preservation of the aspect ratio.
    Args:
        - size (sequence[int])
        - interpolation (InterpolationMode)
        - crop_pct (float, optional): optional extra cropping not related to preservation of aspect ratio.
    """
    
    def __init__(self, size, interpolation, crop_pct=None):
        scale_size = size if not crop_pct else tuple(int(s / crop_pct) for s in size)
        self.resize = ResizeKeepRatio(scale_size, interpolation)
        self.target_size = size
        self.interpolation = interpolation
        self.crop_pct = crop_pct
    
    def __call__(self, img):
        img = self.resize(img)
        img = F.center_crop(img, self.target_size)
        return img

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.target_size}, interpolation={self.interpolation.value}, crop_pct={self.crop_pct})"


class ToNormalizedTensor:

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, img):
        img = F.to_tensor(img)
        img = F.normalize(img, self.mean, self.std)
        return img
    
    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


def create_augmentation(
        img_size,
        mean,
        std,
        interpolation=None,
        crop_pct=None,
        crop_scale=None,
        crop_ratio=None,
        hflip=None,
        vflip=None,
        aa_config=None,
        aa_interpolation=None,
        color_jitter=None,
        re_prob=None,
        re_value=None,
):
    """
    At the minimum, performs ToNormalizedTensor.
    Everything else is optional.
    Args:
        - img_size (sequence[int])
        - mean (sequence[float])
        - std (sequence[float])
        - interpolation (str, optional)
        - crop_pct (float, optional)
        - crop_scale (sequence[float], optional)
        - crop_ratio (sequence[float], optional)
        - hflip (float, optional)
        - vflip(float, optional)
        - aa_config (str, optional)
        - aa_interpolation (str, optional)
        - color_jitter (float, sequence[float], optional)
        - re_prob (float, optional)
        - re_value (int, sequence[int] or str, optional):
            - if sequence[int]: [R G B]
            - if str: "random"
    """
    res = []

    # Resize & Crop
    assert utils.all_none_or_not(crop_scale, crop_ratio)
    if interpolation:
        interpolation = F.InterpolationMode(interpolation)
        if crop_scale: res += [T.RandomResizedCrop(img_size, scale=crop_scale, ratio=crop_ratio, interpolation=interpolation)]
        else: res += [ResizeCenterCrop(img_size, interpolation, crop_pct)]

    # Flip
    if hflip: res += [T.RandomHorizontalFlip(hflip)]
    if vflip: res += [T.RandomVerticalFlip(vflip)]

    # AutoAugment
    assert utils.all_none_or_not(aa_config, aa_interpolation)
    disable_color_jitter = False
    if aa_config:
        # Color jitter is disabled if AA/RA on.
        disable_color_jitter = not '3a' in aa_config
        aa_params = {
            'translate_const': int(0.45 * min(img_size)),
            'img_mean': tuple(round(255 * m) for m in mean),
            'interpolation': str_to_pil_interp(aa_interpolation),
        }
        if aa_config.startswith('rand'):
            res += [rand_augment_transform(aa_config, aa_params)]
        elif aa_config.startswith('augmix'):
            aa_params['translate_pct'] = 0.3
            res += [augment_and_mix_transform(aa_config, aa_params)]
        else:
            res += [auto_augment_transform(aa_config, aa_params)]

    # Color Jitter
    # Enabled when not using AA.
    if color_jitter and not disable_color_jitter:
        if isinstance(color_jitter, (list, tuple)):
            # Color jitter should be a 3-tuple/list for brightness/contrast/saturation
            # or 4 if also augmenting hue.
            assert len(color_jitter) in (3, 4)
        else:
            # If it's a scalar, duplicate for brightness, contrast, and saturation, no hue.
            color_jitter = (color_jitter,) * 3
        res += [T.ColorJitter(*color_jitter)]

    # Normalize
    res += [ToNormalizedTensor(mean, std)]

    # Random Erasing
    assert utils.all_none_or_not(re_prob, re_value)
    if re_prob: res += [T.RandomErasing(re_prob, value=re_value)]

    return T.Compose(res)