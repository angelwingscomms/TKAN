from .model import init_tkan as _init_tkan


def init_tkan(input_dim, hidden, sub, key, output_dim=1, use_attention=False, attn_dim=64):
    return _init_tkan(
        input_dim,
        hidden,
        sub,
        key,
        output_dim=output_dim,
        use_attention=use_attention,
        attn_dim=attn_dim,
    )
