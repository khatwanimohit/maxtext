import contextlib
import functools

import os
from absl import flags
import numpy as np

import jax
from jax import lax
from jax import random
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

partial = functools.partial

def make_causal_mask(seq_len):
  idx = np.arange(seq_len, dtype=np.int32)
  return (idx[:, None] >= idx[None, :]).astype(np.int32)

def make_local_attention_mask(seq_len, window_size, *, offset = 0):
  idx = np.arange(seq_len, dtype=np.int32)
  mask = np.ones((seq_len, seq_len), dtype=np.int32)
  left, right = window_size
  if left is not None:
    mask = mask & (idx[:, None] - left + offset <= idx[None, :])
  if right is not None:
    mask = mask & (idx[:, None] + right + offset >= idx[None, :])
  return mask

def make_random_mask(seq_len, sparsity, seed: int = 0):
  np.random.seed(seed)
  shape = (seq_len, seq_len)
  return np.random.binomial(n=1, p=1 - sparsity, size=shape).astype(np.int32)


def _process_mask(mask, block_shape):
  # Processes numpy masks for use with the Pallas kernel
  *shape_prefix, m, n = mask.shape
  bm, bn = block_shape
  tm, tn = m // bm, n // bn

  block_mask = np.zeros((*shape_prefix, tm, tn), dtype=mask.dtype)
  mask_load_coords, data_load_coords = [], []
  partial_mask = []
  for idx in np.ndindex(*block_mask.shape):
    *prefix, i, j = idx
    indexer = (*prefix, slice(i * bm, (i + 1) * bm), slice(j * bn, (j + 1) * bn))
    chunk = mask[indexer]
    has_nonzero = chunk.any()
    all_nonzero = chunk.all()
    all_zero = not chunk.any()
    if has_nonzero:
      data_load_coords.append((*prefix, i, j))
      if not all_nonzero:
        partial_mask.append(chunk)
        mask_load_coords.append((*prefix, i, j))
        block_mask[(*prefix, i, j)] = 1
      else:
        block_mask[(*prefix, i, j)] = 2
  partial_mask = np.stack(partial_mask, axis=0)
  j_next = np.zeros_like(block_mask)
  mask_next = np.zeros_like(block_mask)
  data_coords_iter = iter(data_load_coords)
  mask_coords_iter = iter(enumerate(mask_load_coords))
  first_j = coord_j = next(data_coords_iter)
  first_m = coord_m = next(mask_coords_iter)
  for idx in np.ndindex(*j_next.shape):
    (*prefix, i, j) = idx
    if idx > coord_j:
      try:
        coord_j = next(data_coords_iter)
      except StopIteration:
        coord_j = first_j
    if idx > coord_m[1]:
      try:
        coord_m = next(mask_coords_iter)
      except StopIteration:
        coord_m = first_m
    j_next[(*prefix, i, j)] = coord_j[2]
    mask_next[(*prefix, i, j)] = coord_m[0]
  return j_next, mask_next, block_mask, partial_mask

def when(condition):
  def _wrapped(f):
    if isinstance(condition, bool):
      if condition:
        f()
    else:
      jax.lax.cond(condition, f, lambda: None)
  return _wrapped

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)

def _splash_attention_fwd(mask, q, k, v, bq: int, bkv: int):
  *_, seq_len, _ = mask.shape
  j_next, m_next, block_mask, partial_mask = _process_mask(mask, (bq, bkv))
  *_, mq, mkv = block_mask.shape

  j_next, m_next, block_mask = j_next.ravel(), m_next.ravel(), block_mask.ravel()

  def next_nonzero(h, i, j, j_next_ref, block_mask_ref, m_next_ref):
    idx = h * mq * mkv + i * mkv + j
    is_nonzero = block_mask_ref[idx] > 0
    should_not_mask = block_mask_ref[idx] != 1
    next_m = m_next_ref[idx]
    j = jnp.where(is_nonzero, j, j_next_ref[idx])
    next_j = j
    next_m = next_m
    next_j = jnp.where(is_nonzero, j, next_j)
    return next_j, next_m, is_nonzero, should_not_mask

  def flash_attention_kernel(j_next_ref, block_mask_ref, mask_next_ref,
                             q_ref, k_ref, v_ref, mask_ref, m_ref, l_ref,
                             o_ref):
    h, i, j = pl.program_id(0), pl.program_id(1), pl.program_id(2)
    @when(j == 0)
    def init():
      o_ref[...] = jnp.zeros_like(o_ref)
      m_ref[...] = jnp.full_like(m_ref, jnp.NINF)
      l_ref[...] = jnp.zeros_like(l_ref)

    _, next_m, should_run, should_not_mask = next_nonzero(
        h, i, j, j_next_ref, block_mask_ref, mask_next_ref)
    @when(should_run)
    def run():
      q, k, v = q_ref[...], k_ref[...], v_ref[...]
      m_prev, l_prev = m_ref[...], l_ref[...]

      qk = jnp.dot(q, k)
      snm = lax.select(should_not_mask, jnp.array(1), jnp.array(0))
      mask = jnp.bitwise_or(mask_ref[...], jnp.broadcast_to(snm, mask_ref.shape))
      qk = qk + jnp.where(mask, 0., DEFAULT_MASK_VALUE)
      m_curr = qk.max(axis=-1)
      s_curr = jnp.exp(qk - m_curr[..., None])
      l_curr = jax.lax.broadcast_in_dim(s_curr.sum(axis=-1), l_prev.shape, (0,))
      o_curr = jnp.dot(s_curr, v) / l_curr

      m_curr = jax.lax.broadcast_in_dim(m_curr, m_prev.shape, (0,))
      m_next = jnp.maximum(m_prev, m_curr)
      alpha = jnp.exp(m_prev - m_next)
      beta = jnp.exp(m_curr - m_next)
      l_next = alpha * l_prev + beta * l_curr
      l_next_safe = jnp.where(l_next == 0., 1., l_next)

      m_ref[...], l_ref[...] = m_next, l_next_safe
      o_ref[...] =  (l_prev * alpha * o_ref[...] + l_curr * beta * o_curr) / l_next_safe

  @jax.jit
  def flash_attention(mask_info, q, k, v):
    num_heads, seq_len, head_dim = q.shape
    j_next, block_mask, m_next, partial_mask = mask_info

    def q_index_map(h, i, j, j_next_ref, block_mask_ref, mask_next_ref):
      return h, i, 0

    def k_index_map(h, i, j, j_next_ref, block_mask_ref, mask_next_ref):
      next_j, *_ = next_nonzero(h, i, j, j_next_ref, block_mask_ref,
                                mask_next_ref)
      return h, 0, next_j

    def v_index_map(h, i, j, j_next_ref, block_mask_ref, mask_next_ref):
      next_j, *_ = next_nonzero(h, i, j, j_next_ref, block_mask_ref,
                                mask_next_ref)
      return h, next_j, 0

    def mask_index_map(h, i, j, j_next_ref, block_mask_ref, mask_next_ref):
      _, next_m, *_ = next_nonzero(h, i, j, j_next_ref, block_mask_ref,
                                   mask_next_ref)
      return next_m, 0, 0

    k = k.swapaxes(-1, -2)
    return pl.pallas_call(
        flash_attention_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=3,
          in_specs=[
            pl.BlockSpec(q_index_map, (None, bq, head_dim)),
            pl.BlockSpec(k_index_map, (None, head_dim, bkv)),
            pl.BlockSpec(v_index_map, (None, bkv, head_dim)),
            pl.BlockSpec(mask_index_map, (None, bq, bkv)),
          ],
          out_specs=[
            pl.BlockSpec(lambda h, i, j, *_: (0, 0), (bq, head_dim)),
            pl.BlockSpec(lambda h, i, j, *_: (0, 0), (bq, head_dim)),
            pl.BlockSpec(q_index_map, (None, bq, head_dim)),
          ],
          grid=(num_heads, seq_len // bq, seq_len // bkv),
        ),
        mosaic_params=dict(
          dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        out_shape=[
            jax.ShapeDtypeStruct((seq_len, head_dim), q.dtype),  # l
            jax.ShapeDtypeStruct((seq_len, head_dim), q.dtype),  # m
            q,
        ],
    )(j_next, block_mask, m_next, q, k, v, partial_mask)[2]
  return jax.vmap(partial(flash_attention, (jnp.array(j_next), jnp.array(block_mask),
                                             jnp.array(m_next), jnp.array(partial_mask))))

@partial(jax.custom_jvp, nondiff_argnums = (4, 5))
def _splash_attention_custom(mask, q, k, v, bq, bkv):
  return _splash_attention_fwd(
    mask, q, k, v, bq, bkv, 
  )

def _splash_attention_bwd(bq, bkv, res, do_input):
  import pdb;pdb.set_trace()

_splash_attention_custom.defjvp(_splash_attention_fwd, _splash_attention_bwd)


def splash_attention(mask, q, k, v, bq = 512, bkv = 1024):
  return _splash_attention_custom(mask, q, k, v, bq, bkv)

