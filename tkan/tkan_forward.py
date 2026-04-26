from .model import tkan_fwd as _tkan_fwd, tkan_sequence as _tkan_sequence


def tkan_fwd(params, x, hidden=100, sub=20):
    return _tkan_fwd(params, x)


def tkan_sequence(params, x, hidden=100, sub=20):
    return _tkan_sequence(params, x)
