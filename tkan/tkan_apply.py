import jax
import jax.numpy as jnp
from functools import partial
from .tkan_forward import tkan_fwd


@partial(jax.jit, static_argnames=['hidden'])
def tkan_apply(params, x, hidden=100):
    logits = jnp.dot(tkan_fwd(params, x, hidden), params['dense_w']) + params['dense_b']
    if logits.shape[-1] == 1:
        return jax.nn.sigmoid(logits)
    return jax.nn.softmax(logits, axis=-1)
