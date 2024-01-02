"""Base layers."""

from flax import linen as nn
import jax
import jax.numpy as jnp


class Einsum(nn.Module):
  shape: tuple[int, ...]

  @nn.compact
  def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
    w = self.param('w', nn.initializers.zeros_init(), self.shape)
    return jnp.einsum(eqn, x, w)


class RMSNorm(nn.Module):

  @nn.compact
  def __call__(self, x):
    scale = self.param('scale', nn.initializers.zeros_init(), (x.shape[-1]))
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))
    normed_inputs = normed_inputs * (1 + scale)
    return normed_inputs