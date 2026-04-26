import jax
import jax.numpy as jnp


def uses_attention(params):
    return 'attn_v' in params


def init_tkan(input_dim, hidden, sub, key, output_dim=1, use_attention=False, attn_dim=64):
    if use_attention:
        keys = jax.random.split(key, 12)
        return {
            'wx': jax.random.normal(keys[0], (input_dim, hidden * 3)) * 0.01,
            'uh': jax.random.normal(keys[1], (hidden, hidden * 3)) * 0.01,
            'bias': jnp.zeros((hidden * 3,)),
            'sub_wx': jax.random.normal(keys[2], (input_dim, sub)) * 0.01,
            'sub_wh': jax.random.normal(keys[3], (sub, sub)) * 0.01,
            'sub_k': jax.random.normal(keys[4], (sub * 2,)) * 0.01,
            'agg_w': jax.random.normal(keys[5], (sub, hidden)) * 0.01,
            'agg_b': jnp.zeros((hidden,)),
            'feat_wx': jax.random.normal(keys[6], (input_dim, input_dim)) * 0.01,
            'feat_wh': jax.random.normal(keys[7], (hidden, input_dim)) * 0.01,
            'feat_b': jnp.zeros((input_dim,)),
            'attn_w1': jax.random.normal(keys[8], (hidden, attn_dim)) * 0.01,
            'attn_w2': jax.random.normal(keys[9], (hidden, attn_dim)) * 0.01,
            'attn_v': jax.random.normal(keys[10], (attn_dim, 1)) * 0.01,
            'dense_w': jax.random.normal(keys[11], (hidden * 2, output_dim)) * 0.01,
            'dense_b': jnp.zeros((output_dim,)),
        }

    keys = jax.random.split(key, 7)
    return {
        'wx': jax.random.normal(keys[0], (input_dim, hidden * 3)) * 0.3,
        'uh': jax.random.normal(keys[1], (hidden, hidden * 3)) * 0.3,
        'bias': jnp.zeros((hidden * 3,)),
        'sub_wx': jax.random.normal(keys[2], (input_dim, sub)) * 0.2,
        'sub_wh': jax.random.normal(keys[3], (sub, sub)) * 0.2,
        'sub_k': jax.random.normal(keys[4], (sub * 2,)) * 0.2,
        'agg_w': jax.random.normal(keys[5], (sub, hidden)) * 0.3,
        'agg_b': jnp.zeros((hidden,)),
        'dense_w': jax.random.normal(keys[6], (hidden, output_dim)) * 0.3,
        'dense_b': jnp.zeros((output_dim,)),
    }


@jax.jit
def input_feature_attention(params, x_t, h_prev):
    energy = jnp.tanh(
        jnp.dot(x_t, params['feat_wx']) +
        jnp.dot(h_prev, params['feat_wh']) +
        params['feat_b']
    )
    feature_weights = jax.nn.softmax(energy, axis=-1)
    return x_t * feature_weights * x_t.shape[-1]


@jax.jit
def tkan_cell(params, h, c, x, sub_s):
    hidden = h.shape[-1]
    sub = sub_s.shape[-1]

    gates = jnp.dot(x, params['wx']) + jnp.dot(h, params['uh']) + params['bias']
    i = jax.nn.sigmoid(gates[:, :hidden])
    f = jax.nn.sigmoid(gates[:, hidden:hidden * 2])
    cg = jnp.tanh(gates[:, hidden * 2:])
    c_new = f * c + i * cg

    sub_o = jnp.tanh(jnp.dot(x, params['sub_wx']) + jnp.dot(sub_s, params['sub_wh']))
    kh = params['sub_k'][:sub]
    kx = params['sub_k'][sub:]
    new_sub = kh * sub_o + kx * sub_s

    agg = jnp.dot(sub_o, params['agg_w']) + params['agg_b']
    o = jax.nn.sigmoid(agg)
    h_new = o * jnp.tanh(c_new)
    return h_new, c_new, new_sub


def _init_state(params, batch_size):
    hidden = params['uh'].shape[0]
    sub = params['sub_wh'].shape[0]
    h0 = jnp.zeros((batch_size, hidden))
    c0 = jnp.zeros((batch_size, hidden))
    sub0 = jnp.zeros((batch_size, sub))
    return h0, c0, sub0


def _step_fn(params, use_attention):
    def step(carry, x_t):
        h, c, sub_s = carry
        if use_attention:
            x_t = input_feature_attention(params, x_t, h)
        h_new, c_new, sub_new = tkan_cell(params, h, c, x_t, sub_s)
        return (h_new, c_new, sub_new), h_new

    return step


@jax.jit
def tkan_fwd(params, x):
    step = _step_fn(params, uses_attention(params))
    x_steps = jnp.transpose(x, (1, 0, 2))
    _, hs = jax.lax.scan(step, _init_state(params, x.shape[0]), x_steps)
    return hs[-1]


@jax.jit
def tkan_sequence(params, x):
    step = _step_fn(params, uses_attention(params))
    x_steps = jnp.transpose(x, (1, 0, 2))
    _, hs = jax.lax.scan(step, _init_state(params, x.shape[0]), x_steps)
    return jnp.transpose(hs, (1, 0, 2))


@jax.jit
def bahdanau_temporal_attention(params, hs, h_last):
    h_last_expanded = jnp.expand_dims(h_last, axis=1)
    score_hs = jnp.dot(hs, params['attn_w1'])
    score_last = jnp.dot(h_last_expanded, params['attn_w2'])
    energy = jnp.tanh(score_hs + score_last)
    attention_weights = jnp.dot(energy, params['attn_v'])
    alpha = jax.nn.softmax(attention_weights, axis=1)
    context_vector = jnp.sum(alpha * hs, axis=1)
    return context_vector, alpha


def _to_probabilities(logits):
    if logits.shape[-1] == 1:
        return jax.nn.sigmoid(logits)
    return jax.nn.softmax(logits, axis=-1)


@jax.jit
def _tkan_apply_base(params, x):
    logits = jnp.dot(tkan_fwd(params, x), params['dense_w']) + params['dense_b']
    return _to_probabilities(logits)


@jax.jit
def _tkan_apply_attention(params, x):
    hs = tkan_sequence(params, x)
    h_last = hs[:, -1, :]
    context, temporal_weights = bahdanau_temporal_attention(params, hs, h_last)
    combined = jnp.concatenate([context, h_last], axis=-1)
    logits = jnp.dot(combined, params['dense_w']) + params['dense_b']
    return _to_probabilities(logits), temporal_weights


def tkan_apply(params, x):
    if uses_attention(params):
        preds, _ = _tkan_apply_attention(params, x)
        return preds
    return _tkan_apply_base(params, x)


def tkan_apply_with_attention(params, x):
    if not uses_attention(params):
        raise ValueError('Attention weights are only available when use_attention is enabled.')
    return _tkan_apply_attention(params, x)
