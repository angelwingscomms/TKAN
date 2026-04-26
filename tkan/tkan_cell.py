from .model import tkan_cell as _tkan_cell


def tkan_cell(params, h, c, x, sub_s, hidden=100, sub=20):
    return _tkan_cell(params, h, c, x, sub_s)
