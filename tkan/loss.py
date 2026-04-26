import jax
import jax.numpy as jnp
from functools import partial
from .model import tkan_apply


@partial(jax.jit)
def classification_loss(params, x, y):
    preds = tkan_apply(params, x)
    eps = 1e-8
    if preds.shape[-1] == 1:
        return -jnp.mean(y * jnp.log(preds + eps) + (1 - y) * jnp.log(1 - preds + eps))
    return -jnp.mean(jnp.sum(y * jnp.log(preds + eps), axis=-1))


@partial(jax.jit)
def accuracy(preds, y):
    if preds.shape[-1] == 1:
        return jnp.mean((preds > 0.5) == (y > 0.5))
    return jnp.mean(jnp.argmax(preds, axis=-1) == jnp.argmax(y, axis=-1))


def bce_loss(params, x, y):
    return classification_loss(params, x, y)


def eval_loss(params, x, y, batch_size=128):
    total, count = 0.0, 0
    for i in range(0, len(x), batch_size):
        bx, by = x[i:i+batch_size], y[i:i+batch_size]
        total += float(classification_loss(params, bx, by))
        count += 1
    return total / count if count > 0 else 0.0
