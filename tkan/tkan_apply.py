from .model import tkan_apply as _tkan_apply, tkan_apply_with_attention as _tkan_apply_with_attention


def tkan_apply(params, x, hidden=100, sub=20):
    return _tkan_apply(params, x)


def tkan_apply_with_attention(params, x, hidden=100, sub=20):
    return _tkan_apply_with_attention(params, x)
