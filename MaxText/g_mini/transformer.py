"""G_mini transformer."""

import dataclasses

from flax import linen as nn
from g_mini import layers
from g_mini import modules
from g_mini import positional_embeddings
import jax
import jax.numpy as jnp


@dataclasses.dataclass
class TransformerConfig:
  """Configuration for the G_mini transformer."""

  num_layers: int
  num_embed: int
  embed_dim: int
  hidden_dim: int
  num_heads: int
  head_dim: int


class Transformer(nn.Module):
  """G_mini transformer."""

  config: TransformerConfig

  def setup(self):
    self.embedder = modules.Embedder(
        vocab_size=self.config.num_embed,
        embed_dim=self.config.embed_dim,
    )
    self.blocks = [
        modules.Block(
            name=f'layer_{i}',
            num_heads=self.config.num_heads,
            embed_dim=self.config.embed_dim,
            head_dim=self.config.head_dim,
            hidden_dim=self.config.hidden_dim,
        )
        for i in range(self.config.num_layers)
    ]
    self.final_norm = layers.RMSNorm()

  def __call__(
      self,
      last_token: int,
      current_token_position: int,
      cache: dict[str, modules.LayerCache],
      attention_mask: jax.Array,
      time_step: int,
  ) -> tuple[jax.Array, dict[str, modules.LayerCache]]:
    input_emb = self.embedder.encode(last_token)
    x = positional_embeddings.add_positional_embedding(
        input_emb, current_token_position
    )
    x = jnp.expand_dims(x, 0)

    for i, block in enumerate(self.blocks):
      layer_name = f'layer_{i}'
      cache[layer_name], x = block(
          x,
          current_token_position,
          cache[layer_name],
          attention_mask,
          time_step,
      )

    x = self.final_norm(x)
    logits = self.embedder.decode(x)

    return logits, cache
