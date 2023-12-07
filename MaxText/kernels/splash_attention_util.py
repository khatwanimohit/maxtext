import splash_attention_mask as mask_lib

def dynamic_causal_mask0(
    q_seq_len: int, kv_seq_len: int
) -> mask_lib.DynamicCausalMaskBounds:
  return mask_lib.DynamicCausalMaskBounds(
      causal_offset=0,
      min_q=0,
      max_q=q_seq_len,
      min_kv=0,
      max_kv=kv_seq_len,
  )


def dynamic_causal_mask1(
    q_seq_len: int, kv_seq_len: int
) -> mask_lib.DynamicCausalMaskBounds:
  return mask_lib.DynamicCausalMaskBounds(
      causal_offset=kv_seq_len // 4,
      min_q=0,
      max_q=q_seq_len,
      min_kv=0,
      max_kv=kv_seq_len,
  )


def dynamic_causal_mask2(
    q_seq_len: int, kv_seq_len: int
) -> mask_lib.DynamicCausalMaskBounds:
  return mask_lib.DynamicCausalMaskBounds(
      causal_offset=kv_seq_len // 4,
      min_q=0,
      max_q=3 * q_seq_len // 4,
      min_kv=0,
      max_kv=kv_seq_len,
  )


def dynamic_causal_mask3(
    q_seq_len: int, kv_seq_len: int
) -> mask_lib.DynamicCausalMaskBounds:
  return mask_lib.DynamicCausalMaskBounds(
      causal_offset=kv_seq_len // 4,
      min_q=0,
      max_q=3 * q_seq_len // 4,
      min_kv=kv_seq_len // 4,
      max_kv=kv_seq_len,
  )


DYNAMIC_CAUSAL_MASK_BOUNDS_GENERATORS = (
    dynamic_causal_mask0,
    dynamic_causal_mask1,
    dynamic_causal_mask2,
    dynamic_causal_mask3,
)