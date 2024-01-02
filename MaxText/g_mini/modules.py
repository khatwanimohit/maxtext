"""Transformer sub-modules.

TODO(g_mini): add comments and docstrings.
"""

from typing import cast

from flax import linen as nn
from g_mini import layers
from g_mini import positional_embeddings
import jax
import jax.numpy as jnp

K_MASK = -2.3819763e38  # Set to a large negative number.
LayerCache = dict[str, jax.Array]


def init_layer_cache(
    cache_size: int, num_heads: int, head_dim: int
) -> LayerCache:
  return {
      'v': jnp.zeros((cache_size, num_heads, head_dim), dtype=jnp.float32),
      'k': jnp.zeros((cache_size, num_heads, head_dim), dtype=jnp.float32),
  }


class Embedder(nn.Module):
  """Embedder module."""

  vocab_size: int
  embed_dim: int

  def setup(self):
    self.input_embedding_table = self.param(
        'input_embedding',
        nn.initializers.zeros_init(),
        (self.vocab_size, self.embed_dim),
    )

  def encode(self, x: int) -> jax.Array:
    x = self.input_embedding_table[(x,)]
    x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
    return x

  def decode(self, x: jax.Array) -> jax.Array:
    return jnp.dot(x, self.input_embedding_table.T)


class Attention(nn.Module):
  """Attention module."""

  num_heads: int
  features: int
  head_dim: int

  def setup(self):
    self.qkv_einsum = layers.Einsum(
        shape=(3, self.num_heads, self.features, self.head_dim),
    )
    self.attn_vec_einsum = layers.Einsum(
        shape=(self.num_heads, self.head_dim, self.features),
    )

  def __call__(
      self,
      x: jax.Array,
      segment_pos: int,
      cache: LayerCache,
      attn_mask: jax.Array,
      time_step: int,
  ) -> tuple[LayerCache, jax.Array]:
    query_proj, key_proj, value_proj = self.qkv_einsum('TD,SNDH->STNH', x[0])

    query_proj = positional_embeddings.apply_rope(
        query_proj,
        segment_pos,
        head_dim=self.head_dim,
    )
    query_scaled = query_proj * self.head_dim**-0.5

    key_proj = positional_embeddings.apply_rope(
        key_proj,
        segment_pos,
        head_dim=self.head_dim,
    )

    cache['v'] = cache['v'].at[[time_step], :, :].set(value_proj)
    cache['k'] = cache['k'].at[[time_step], :, :].set(key_proj)

    logits = jnp.einsum('BNH,SNH->BNS', query_scaled, cache['k'])
    logits = logits.astype(jnp.float32)

    padded_logits = jnp.where((attn_mask >= K_MASK * 0.5), logits, K_MASK)
    probs = jax.nn.softmax(padded_logits, axis=-1).astype(cache['k'].dtype)

    encoded = jnp.einsum('BNS,SNH->BNH', probs, cache['v'])
    attn_output = self.attn_vec_einsum('ANH,NHD->AD', encoded)

    return cache, attn_output


class FeedForward(nn.Module):
  """Feed forward module."""

  features: int
  hidden_dim: int

  @nn.compact
  def __call__(self, x):
    w_gating = self.param(
        'gating_einsum',
        nn.initializers.zeros_init(),
        ((2, self.features, self.hidden_dim)),
    )
    ff_gate = jnp.dot(x, w_gating[0])
    gate_value = nn.gelu(ff_gate)

    ff1 = jnp.dot(x, w_gating[1])
    activations = gate_value * ff1

    w_linear = self.param(
        'linear',
        nn.initializers.zeros_init(),
        (self.hidden_dim, self.features),
    )
    outputs = jnp.dot(activations, w_linear)

    return outputs


class Block(nn.Module):
  """Transformer block."""

  num_heads: int
  embed_dim: int
  head_dim: int
  hidden_dim: int

  def setup(self):
    self.pre_attention_norm = layers.RMSNorm()
    self.attn = Attention(
        num_heads=self.num_heads,
        features=self.embed_dim,
        head_dim=self.head_dim,
    )
    self.pre_ffw_norm = layers.RMSNorm()
    self.mlp = FeedForward(features=self.embed_dim, hidden_dim=self.hidden_dim)

  def __call__(
      self,
      x: jax.Array,
      segment_pos: int,
      cache: LayerCache,
      attn_mask: jax.Array,
      time_step: int,
  ):
    inputs_normalized = self.pre_attention_norm(x)
    cache, attn_output = self.attn(
        inputs_normalized, segment_pos, cache, attn_mask, time_step
    )
    attn_output += x
    residual = attn_output
    attn_output = self.pre_ffw_norm(attn_output)
    outputs = self.mlp(attn_output)
    outputs = residual + outputs
    return cache, outputs