from .patchify import Patchify


def get_patchify(patch_size):
    """shape: [b c N1 ... Nd] -> [b n1 ... nd d]"""
    return Patchify(None, None, patch_size, None, False)