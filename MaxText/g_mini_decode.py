"""
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

# pylint: disable=g-bad-todo, abstract-method, consider-using-with
"""Training loop and Decoding of the model."""
import functools
from typing import Sequence

import os
from absl import app
from flax.linen import partitioning as nn_partitioning
import numpy as np
import seqio
import pyconfig
import max_utils

import checkpointing

import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh
from jax.experimental.compilation_cache import compilation_cache as cc

from g_mini import transformer
from g_mini import modules
from g_mini import params as params_lib

import max_logging

from input_pipeline import create_data_iterator_with_tokenizer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
cc.initialize_cache(os.path.expanduser("~/jax_cache"))

Transformer = transformer.Transformer

def decode_tokens(toks, tokenizer, eos_id):
  if np.argmax(toks == eos_id) > 0:
    valid_toks = toks[:np.argmax(toks == eos_id)]
  else:
    valid_toks = toks
    valid_toks[-1] = eos_id

  valid_toks = valid_toks.astype(np.int32)
  return tokenizer.detokenize(valid_toks).numpy().decode("utf-8"), len(valid_toks)


def encode_strings(strs, max_len, tokenizer):
  tokenized_batch = np.zeros((len(strs), max_len), np.int32)
  for i, s in enumerate(strs):
    toks = tokenizer.tokenize(s).numpy()
    # Remove EOS token in prompt.
    tokenized_batch[i, :toks.shape[0]-1] = toks[:-1]
  return tokenized_batch

def decode_loop(config, state=None):
  """Decoding loop for the Transformer model."""
  # checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(config.checkpoint_dir,
  #                                                                    config.enable_checkpointing,
  #                                                                    config.async_checkpointing,
  #                                                                    config.save_period)
  rng = random.PRNGKey(0)

  # Mesh definition
  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)

  # TODO(mohitkhatwani) : Change this to gcs tokenizer path
  # vocab = seqio.SentencePieceVocabulary(
  #       '/placer/prod/home/gemini-data-access/tokenizers/v1.1/gemini_bpe_256k_v5_no_tags.model'
  # )
  _, vocab = create_data_iterator_with_tokenizer(config, mesh)
  transformer_config = transformer.TransformerConfig(
      num_layers=18,
      num_embed=vocab.vocab_size().numpy() + 128,  # 128 for padding
      embed_dim=2176,
      hidden_dim=17408,
      num_heads=8,
      head_dim=256,
  )
  model = Transformer(config = transformer_config)
  state, state_mesh_annotations = max_utils.setup_decode_state(
    model, config, rng, mesh, None
  )
  print(state)


def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  os.environ["TFDS_DATA_DIR"] = pyconfig.config.dataset_path
  os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
  decode_loop(pyconfig.config)


if __name__ == "__main__":
  app.run(main)