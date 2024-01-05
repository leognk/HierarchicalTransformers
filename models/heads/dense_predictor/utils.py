from .unpatch import Unpatch


def get_unpatch(patch_size):
    """shape: [b n1 ... nd d] -> [b c N1 ... Nd]"""
    return Unpatch(None, None, patch_size, None)