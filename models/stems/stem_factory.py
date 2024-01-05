from . import linear
from . import conv
from . import single_conv


def create_stem(name, in_channels, in_size, patch_size, emb_dim, add_sin_posemb, *args, **kwargs):
    """
    stem shape: [b c N1 ... Nd] -> [b n1 ... nd d]
        where c=in_channels, N=(n p), p=patch_size, d=emb_dim
    """
    models = {
        'linear': linear.Linear,
        'conv': conv.Conv,
        'single_conv': single_conv.SingleConv,
    }
    model = models[name](in_channels, in_size, patch_size, emb_dim, add_sin_posemb, *args, **kwargs)
    return model