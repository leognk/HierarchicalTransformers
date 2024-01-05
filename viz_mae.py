import torch
from datasets import create_dataset, create_transform
from models import create_ssl
import os
from einops import rearrange, pack, unpack, repeat
from datasets.utils import plot_img
import logger


ckpt_root = "ssl/in1k/mae"

ckpt_name = "vit/exp1/run1/ckpt-800.pt"
model_name = "mae_vit"
model_config = "imagenet/base-16.yaml"

# ckpt_name = "sft-s2s/exp1/run2/ckpt-100.pt"
# model_name = "mae_sft_s2s"
# model_config = "imagenet/no_rec/base-4.yaml"


torch.random.manual_seed(0)

dataset = create_dataset("imagenet", "full.yaml", train=False)
dataset.transform = create_transform(dataset, "eval.yaml")

model = create_ssl(model_name, model_config, dataset.channels, dataset.transform.img_size)
model.cuda()
model.eval()

# load from ckpt
ckpt, _ = logger.Run.load_ckpt(os.path.join(ckpt_root, ckpt_name), root="akiu/runs", map_location='cuda:0')
sd = ckpt['model']['params']
model.load_state_dict(sd)

plt_img = lambda im: plot_img(im, mean=dataset.mean, std=dataset.std, figsize=1)

p1, p2 = model.patch_size

def patch(x):
    x = rearrange(x, 'c (n1 p1) (n2 p2) -> n1 n2 (p1 p2 c)', p1=p1, p2=p2)
    x, ps = pack([x], '* d')
    return x, ps

def unpatch(x, ps):
    [x] = unpack(x, ps, '* d')
    x = rearrange(x, 'n1 n2 (p1 p2 c) -> c (n1 p1) (n2 p2)', p1=p1, p2=p2)
    return x

with torch.no_grad():
    for i in range(0, 10):
        im, _ = dataset[i]
        im = im.cuda()

        x = im.unsqueeze(0)
        _, _, pred, masks = model(x)
        pred = pred[0]
        
        # normalize im patches
        im, ps = patch(im)
        mean = im.mean(dim=-1, keepdim=True)
        var = im.var(dim=-1, keepdim=True)
        im = (im - mean) / (var + 1e-5).sqrt()
        im = unpatch(im, ps)

        # masked image
        mim = im.clone()
        mim, ps = patch(mim)
        zero = -torch.tensor(dataset.mean) / torch.tensor(dataset.std)
        zero = zero.cuda()
        mim[masks[0].type(torch.bool)] = repeat(zero, 'c -> (p1 p2 c)', p1=p1, p2=p2)
        mim = unpatch(mim, ps)

        plt_img(im)
        plt_img(mim)
        plt_img(pred)