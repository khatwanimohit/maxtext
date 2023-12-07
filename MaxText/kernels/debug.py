
import functools
from typing import Any, Callable, NamedTuple, Sequence, Tuple

import jax
import numpy as np
import splash_try as splash
import splash_attention_mask as mask_lib

NamedSharding = jax.sharding.NamedSharding
PartitionSpec = jax.sharding.PartitionSpec

class AttentionData(NamedTuple):
  q: jax.Array
  k: jax.Array
  v: jax.Array
  segment_ids: splash.SegmentIds
  doutput: jax.Array
  dynamic_causal_mask_bounds: mask_lib.DynamicCausalMaskBounds | None


class AttentionDataSharding(NamedTuple):
  q: PartitionSpec
  k: PartitionSpec
  v: PartitionSpec
  segment_ids: splash.SegmentIds
  doutput: PartitionSpec
  dynamic_causal_mask_bounds: mask_lib.DynamicCausalMaskBounds | None



def _make_segment_ids(
    q_seq_len: int,
    kv_seq_len: int,
    segment_count: int,
    batch_size: int | None = None,
) -> splash.SegmentIds:
  assert q_seq_len == kv_seq_len

  seq_len = q_seq_len
  segment_size = seq_len // segment_count
  rem = seq_len % segment_count
  segment_sizes = [segment_size] * segment_count
  segment_sizes[-1] += rem
  segment_ids_array = np.repeat(np.arange(segment_count), segment_sizes)
  assert len(segment_ids_array) == seq_len
  if batch_size is not None:
    segment_ids_array = np.broadcast_to(
        np.expand_dims(segment_ids_array, 0), (batch_size, seq_len)
    )
  return splash.SegmentIds(segment_ids_array, segment_ids_array)


if __name__ == "__main__":
  _make_segment_ids(1024, 1024, 3, 8)