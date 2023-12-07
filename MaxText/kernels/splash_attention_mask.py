"""Mini-mask creation library."""

import dataclasses
import functools
from typing import Any, Callable, NamedTuple, Protocol, Sequence, Tuple
import jax
from jax import util as jax_util
import jax.numpy as jnp
import numpy as np


class Mask:
  """A base class for splash attention masks."""

  @property
  def shape(self) -> Tuple[int, ...]:
    raise NotImplementedError

  def __getitem__(self, idx) -> np.ndarray:
    raise NotImplementedError

  def __bool__(self) -> bool:
    raise NotImplementedError(
        'Conversion to bool is unsupported. Could be caused by using logical'
        ' instead of bitwise operations on masks.'
    )

  def __or__(self, other: 'Mask') -> 'Mask':
    if self.shape != other.shape:
      raise ValueError(
          f'Invalid shape for other: {other.shape}, expected: {self.shape}'
      )
    return LogicalOr(self, other)

  def __and__(self, other: 'Mask') -> 'Mask':
    if self.shape != other.shape:
      raise ValueError(
          f'Invalid shape for other: {other.shape}, expected: {self.shape}'
      )
    return LogicalAnd(self, other)


def make_causal_mask(shape: Tuple[int, int], offset: int = 0) -> np.ndarray:
  """Makes a causal attention mask.

  Args:
    shape: Shape of the 2-dim mask: (q_seq_len, kv_seq_len).
    offset: Offset of q start wrt kv. A positive offset shifts the bottom
      triangle upward, a negative one shifts it downward. A negative offset
      makes the first 'offset' rows of the attention matrix all 0s which leads
      to undefined softmax.

  Returns:
    The causal mask.
  """
  q_seq_len, kv_seq_len = shape
  q_idx = np.arange(q_seq_len, dtype=np.int32)
  kv_idx = np.arange(kv_seq_len, dtype=np.int32)
  return (q_idx[:, None] + offset >= kv_idx[None, :]).astype(np.bool_)


def make_local_attention_mask(
    shape: Tuple[int, int],
    window_size: Tuple[int | None, int | None],
    *,
    offset: int = 0,
) -> np.ndarray:
  """Makes a local attention mask."""
  q_seq_len, kv_seq_len = shape
  q_idx = np.arange(q_seq_len, dtype=np.int32)
  kv_idx = np.arange(kv_seq_len, dtype=np.int32)
  mask = np.ones((q_seq_len, kv_seq_len), dtype=np.bool_)
  left, right = window_size
  if left is not None:
    mask = mask & (q_idx[:, None] - left + offset <= kv_idx[None, :])
  if right is not None:
    mask = mask & (q_idx[:, None] + right + offset >= kv_idx[None, :])
  return mask.astype(np.bool_)


def make_random_mask(
    shape: Tuple[int, int], sparsity: float, seed: int
) -> np.ndarray:
  """Makes a random attention mask."""
  np.random.seed(seed)
  return np.random.binomial(n=1, p=1.0 - sparsity, size=shape).astype(np.bool_)


@dataclasses.dataclass
class LogicalOr(Mask):
  left: Mask
  right: Mask

  def __init__(self, left: Mask, right: Mask):
    if left.shape != right.shape:
      raise ValueError('Masks must have the same shape')
    self.left = left
    self.right = right

  @property
  def shape(self) -> Tuple[int, ...]:
    return self.left.shape

  def __getitem__(self, idx) -> np.ndarray:
    return self.left[idx] | self.right[idx]

  def __hash__(self):
    return hash((type(self),) + (self.left, self.right))


@dataclasses.dataclass
class LogicalAnd(Mask):
  left: Mask
  right: Mask

  def __init__(self, left: Mask, right: Mask):
    if left.shape != right.shape:
      raise ValueError('Masks must have the same shape')
    self.left = left
    self.right = right

  @property
  def shape(self) -> Tuple[int, ...]:
    return self.left.shape

  def __getitem__(self, idx) -> np.ndarray:
    return self.left[idx] & self.right[idx]

  def __hash__(self):
    return hash((type(self),) + (self.left, self.right))


@dataclasses.dataclass
class MultiHeadMask(Mask):
  """Lazy multihead mask, combines multiple lazy masks one per head."""

  masks: Sequence[Mask]

  def __post_init__(self):
    if not self.masks:
      raise ValueError('Unsupported empty tuple of masks')

    shape = self.masks[0].shape
    for mask in self.masks[1:]:
      if shape != mask.shape:
        raise ValueError(
            f'Unexpected mask shape, got: {mask.shape}, expected: {shape}'
        )

    if not all([isinstance(mask, Mask) for mask in self.masks]):
      raise ValueError('masks should be of type Mask')

    if any([isinstance(mask, MultiHeadMask) for mask in self.masks]):
      raise ValueError('Nesting MultiHeadMasks is not supported')

  @property
  def shape(self) -> Tuple[int, ...]:
    return (len(self.masks),) + self.masks[0].shape

  def __getitem__(self, idx) -> np.ndarray:
    if len(idx) != 3:
      raise NotImplementedError(f'Unsupported slice: {idx}')

    head_slice = idx[0]
    if isinstance(head_slice, int):
      assert head_slice >= 0 and head_slice <= len(self.masks)
      return self.masks[head_slice][idx[1:]]
    else:
      slice_masks = [mask[idx[1:]] for mask in self.masks[head_slice]]
      return np.stack(slice_masks)

  def __eq__(self, other: 'MultiHeadMask'):
    if not isinstance(other, type(self)):
      return NotImplemented

    return self.masks == other.masks

  def __hash__(self):
    return hash((type(self),) + tuple(hash(mask) for mask in self.masks))


class InterleaveCallable(Protocol):

  # Leave the first argument unspecified so that the type is compatible with
  # both jax.Array and np.ndarray. Using the type annotation jax.Array |
  # np.ndarray causes __getitem__ to fail type-checking.
  def __call__(self, x: Any, dim: int, shard_count: int):
    ...


class DynamicCausalMaskBounds(NamedTuple):
  """Dynamic bounds for a Causal Mask.

    The mask bounds are described by this figure:

                                KV

                               causal
                     min_kv    offset      max_kv
                          |    |           |
             │000000000000│0000000000000000│000000000000|
             │000000000000│0000000000000000│000000000000|
       min_q │000000000000│1111100000000000│000000000000|
             │000000000000│1111110000000000│000000000000|
  Q          │000000000000│1111111000000000│000000000000|
             │000000000000│1111111100000000│000000000000|
       max_q │000000000000│1111111110000000│000000000000|
             │000000000000000000000000000000000000000000│
             │000000000000000000000000000000000000000000│
  """

  causal_offset: jax.Array | int
  min_q: jax.Array | int
  max_q: jax.Array | int
  min_kv: jax.Array | int
  max_kv: jax.Array | int


def interleave(x: Any, *, dim: int, shard_count: int):
  """Transposes and reshards x in an interleaving way on dim.

  E.g., assuming x is [0, 1, 2, ..., 15]. With shard_count == 4, the result will
  be: [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]

  Args:
    x: The tensor to interleave.
    dim: The dimension to interleave on.
    shard_count: The number of shards on dim.

  Returns:
    The interleaved tensor.
  """
  # Account for negative indices.
  dim = dim % x.ndim
  if shard_count == 1:
    return x
  assert x.shape[dim] % (shard_count * shard_count) == 0, (
      x.shape,
      dim,
      shard_count,
  )
  x = x.reshape(
      x.shape[:dim] + (shard_count, shard_count, -1) + x.shape[dim + 1 :]
  )
  x = x.swapaxes(dim, dim + 1)
  return x.reshape(x.shape[:dim] + (-1,) + x.shape[dim + 3 :])


def evaluate_dynamic_causal_mask(
    q_index,
    kv_index,
    causal_offset,
    min_q,
    max_q,
    min_kv,
    max_kv,
):
  """Return the value of dynamic attention mask at given q and kv index for the given bounds."""

  # pyformat: disable
  operands = [
      min_q <= q_index, q_index < max_q,
      min_kv <= kv_index, kv_index < max_kv,
      q_index + causal_offset >= kv_index,
  ]
  # pyformat: enable
  return functools.reduce(jnp.logical_and, operands)


def make_reference_dynamic_causal_mask(
    q_seq_len: int,
    kv_seq_len: int,
    num_q_heads: int,
    dynamic_mask_bounds: DynamicCausalMaskBounds,
) -> np.ndarray:
  """Return a numpy array corresponding to the dynamic causal mask in input."""
  kv_indices = jnp.broadcast_to(
      jnp.arange(kv_seq_len)[None], (q_seq_len, kv_seq_len)
  )
  q_indices = jnp.broadcast_to(
      jnp.arange(q_seq_len)[:, None], (q_seq_len, kv_seq_len)
  )

  compute_dynamic_mask = functools.partial(
      evaluate_dynamic_causal_mask,
      min_q=dynamic_mask_bounds.min_q,
      max_q=dynamic_mask_bounds.max_q,
      min_kv=dynamic_mask_bounds.min_kv,
      max_kv=dynamic_mask_bounds.max_kv,
      causal_offset=dynamic_mask_bounds.causal_offset,
  )

  vector_mask_function = jax.vmap(jax.vmap(compute_dynamic_mask))

  reference_mask = vector_mask_function(q_indices, kv_indices).astype(jnp.bool_)
  reference_mask = jnp.broadcast_to(
      reference_mask, (num_q_heads, *reference_mask.shape)
  )

  return np.array(reference_mask)


class _ComputableMask(Mask):
  """Superclass for all masks that can be computed inside the kernel using a callable object.

  Attributes:
    _shape: Shape of the 2-dim mask: (q_seq_len, kv_seq_len).
    offset: Offset of q start wrt kv. A positive offset shifts the bottom
      triangle upward, a negative one shifts it downward. A negative offset
      makes the first 'offset' rows of the attention matrix all 0s which leads
      to undefined softmax.
    q_sequence: Indices of Q sequence, which are possibly interleaved.
      q_sequence is reused across __getitem__ calls which is important for
      compile-time performance.
    mask_function: Function used by the SplashAttention kernel to compute the
      mask rather than loading it.
  """

  _shape: Tuple[int, int]
  q_sequence: np.ndarray
  mask_function: Callable[..., Any]

  def __init__(
      self,
      shape: Tuple[int, int],
      mask_function: Callable[..., Any],
      interleave_q: InterleaveCallable | None = None,
      shard_count: int = 1,
  ):
    self._shape = shape
    self.mask_function = mask_function
    q_seq_len = self.shape[0]

    if q_seq_len % (shard_count * shard_count) != 0:
      raise ValueError(
          f'Shard count squared ({shard_count * shard_count}) must'
          f' divide Q seq_len ({self.shape[0]}) evenly.'
      )

    # Save the interleaved q_sequence (rows) to avoid interleaving at every
    # invocation of __getitem__.
    self.q_sequence = np.arange(q_seq_len, dtype=np.int32)
    if interleave_q is not None:
      self.q_sequence = interleave_q(
          x=self.q_sequence, dim=0, shard_count=shard_count
      )

  @property
  def shape(self) -> Tuple[int, ...]:
    return self._shape

  def __getitem__(self, idx) -> np.ndarray:
    if len(idx) != 2:
      raise NotImplementedError(f'Unsupported slice: {idx}')

    q_slice, kv_slice = idx
    if not isinstance(q_slice, slice) or not isinstance(kv_slice, slice):
      raise NotImplementedError(f'Unsupported slice: {idx}')

    q_slice = _fill_slice(q_slice, self.shape[0])
    kv_slice = _fill_slice(kv_slice, self.shape[1])

    rows = self.q_sequence[q_slice]
    cols = np.arange(kv_slice.start, kv_slice.stop)

    return self.mask_function(rows[:, None], cols[None, :])

  def __eq__(self, other: '_ComputableMask'):
    raise NotImplementedError()

  def __hash__(self):
    raise NotImplementedError()


class CausalMask(_ComputableMask):
  """Lazy causal mask, prevents the model from attending to future tokens.

  Attributes:
    offset: Offset of q start wrt kv. A positive offset shifts the bottom
      triangle upward, a negative one shifts it downward. A negative offset
      makes the first 'offset' rows of the attention matrix all 0s which leads
      to undefined softmax.
  """

  offset: int

  def __init__(
      self,
      shape: Tuple[int, int],
      offset: int = 0,
      interleave_q: InterleaveCallable | None = None,
      shard_count: int = 1,
  ):
    self.offset = offset

    def causal_mask_function(q_ids, kv_ids):
      # When evaluating the mask in _process_mask we typically work with numpy
      # array views.
      # Avoid the addition when possible to avoid instantiating an actual array.
      if self.offset == 0:
        return q_ids >= kv_ids
      else:
        return q_ids + self.offset >= kv_ids

    mask_function = causal_mask_function

    super().__init__(
        shape=shape,
        mask_function=mask_function,
        interleave_q=interleave_q,
        shard_count=shard_count,
    )

  def __eq__(self, other: 'CausalMask'):
    if not isinstance(other, type(self)):
      return NotImplemented

    return (
        self.shape == other.shape
        and self.offset == other.offset
        # Hashing the callable interleave_q does not give deterministic results.
        # Instead hash the q_sequence indices. This is more expensive but
        # deterministic.
        and np.array_equal(self.q_sequence, other.q_sequence)
    )

  def __hash__(self):
    return hash((
        type(self),
        self.shape,
        self.offset,
        self.q_sequence.tobytes() if self.q_sequence is not None else None,
    ))


class DynamicMask(_ComputableMask):
  """Fully dynamic mask.

  The value of the mask is determined by the callable in input to to
  constructor.

  Attributes:
    _shape: Shape of the 2-dim mask: (q_seq_len, kv_seq_len).
    q_sequence: Indices of Q sequence, which are possibly interleaved.
      q_sequence is reused across __getitem__ calls which is important for
      compile-time performance.
    mask_function: Function used by the SplashAttention kernel to compute the
      mask rather than loading it.
  """

  def __init__(
      self,
      shape: Tuple[int, int],
      mask_function: Callable[..., Any],
      interleave_q: InterleaveCallable | None = None,
      shard_count: int = 1,
  ):
    super().__init__(
        shape=shape,
        mask_function=mask_function,
        interleave_q=interleave_q,
        shard_count=shard_count,
    )

  def __eq__(self, other: 'DynamicMask'):
    if not isinstance(other, type(self)):
      return NotImplemented

    self_function = jax_util.HashableFunction(self.mask_function, ())
    other_function = jax_util.HashableFunction(other.mask_function, ())

    return (
        self.shape == other.shape
        and self_function == other_function
        and np.array_equal(self.q_sequence, other.q_sequence)
    )

  def __hash__(self):
    return hash((
        type(self),
        self.shape,
        jax_util.HashableFunction(self.mask_function, ()),
        self.q_sequence.tobytes() if self.q_sequence is not None else None,
    ))


class LocalMask(Mask):
  """Lazy local mask, prevents model from attending to tokens outside window.

  Attributes:
    _shape: Shape of the 2-dim mask: (q_seq_len, kv_seq_len).
    window_size: Size of the two sides of the local window (None identifes no
      limit for the given side).
    offset: Offset of q start wrt kv. A positive offset shifts the bottom
      triangle upward, a negative one shifts it downward. A negative offset
      makes the first 'offset' rows of the attention matrix all 0s which leads
      to undefined softmax.
    _q_sequence: When interleaving it contains the interleaved Q sequence, so
      that it can be reused across __getitem__ calls. Important for performance.
  """

  # TODO(amagni): Transform LocalMask into a _ComputableMask.

  _shape: Tuple[int, int]
  window_size: Tuple[int | None, int | None]
  offset: int
  _q_sequence: np.ndarray = None

  def __init__(
      self,
      shape: Tuple[int, int],
      window_size: Tuple[int | None, int | None],
      offset: int,
      interleave_q: InterleaveCallable | None = None,
      shard_count: int = 1,
  ):
    self._shape = shape
    self.window_size = window_size
    self.offset = offset

    if self.shape[0] % (shard_count * shard_count) != 0:
      raise ValueError(
          f'Shard count squared ({shard_count * shard_count}) must'
          f' divide Q seq_len ({self.shape[0]}) evenly.'
      )

    if interleave_q is not None:
      q_seq_len = self.shape[0]
      self._q_sequence = interleave_q(
          x=np.arange(q_seq_len), dim=0, shard_count=shard_count
      )

  @property
  def shape(self) -> Tuple[int, int]:
    return self._shape

  def __getitem__(self, idx) -> np.ndarray:
    if len(idx) != 2:
      raise NotImplementedError(f'Unsupported slice: {idx}')
    q_slice, kv_slice = idx
    if not isinstance(q_slice, slice) or not isinstance(kv_slice, slice):
      raise NotImplementedError(f'Unsupported slice: {idx}')

    q_slice = _fill_slice(q_slice, self.shape[0])
    kv_slice = _fill_slice(kv_slice, self.shape[1])

    if self._q_sequence is None:
      rows = np.arange(q_slice.start, q_slice.stop)
    else:
      rows = self._q_sequence[q_slice]

    cols = np.arange(kv_slice.start, kv_slice.stop)

    left_size, right_size = self.window_size

    if left_size is None and right_size is None:
      return np.ones((rows.shape[0], cols.shape[0]), dtype=np.bool_)
    else:
      expanded_cols = cols[None, :]
      if self.offset != 0:
        expanded_rows = rows[:, None] + self.offset
      else:
        expanded_rows = rows[:, None]
      if left_size is not None and right_size is not None:
        return (expanded_rows <= expanded_cols + left_size) & (
            expanded_cols - right_size <= expanded_rows
        )

      elif left_size is not None and right_size is None:
        return expanded_rows <= expanded_cols + left_size
      else:
        assert left_size is None and right_size is not None
        return expanded_cols - right_size <= expanded_rows

  def __eq__(self, other: 'LocalMask'):
    if not isinstance(other, type(self)):
      return NotImplemented

    return (
        self.shape == other.shape
        and self.window_size == other.window_size
        and self.offset == other.offset
        and np.array_equal(self._q_sequence, other._q_sequence)
    )

  def __hash__(self):
    return hash((
        type(self),
        self.shape,
        self.window_size,
        self.offset,
        self._q_sequence.tobytes() if self._q_sequence is not None else None,
    ))


@dataclasses.dataclass
class NumpyMask(Mask):
  """A mask backed by a dense numpy array."""

  array: np.ndarray

  def __post_init__(self):
    if self.array.ndim != 2:
      raise ValueError('Expected a 2-dim array')

    if self.array.dtype != np.bool_:
      raise ValueError('Mask must be a boolean array')

  @property
  def shape(self) -> Tuple[int, int]:
    return self.array.shape

  def __getitem__(self, idx) -> np.ndarray:
    return self.array[idx]

  def __eq__(self, other: 'NumpyMask'):
    if not isinstance(other, type(self)):
      return NotImplemented

    return np.array_equal(self.array, other.array, equal_nan=True)

  def __hash__(self):
    return hash((type(self), self.array.tobytes()))


def _fill_slice(inp_slice: slice, size: int) -> slice:
  assert inp_slice.step is None or inp_slice.step == 1
  start = 0 if inp_slice.start is None else inp_slice.start
  stop = size if inp_slice.stop is None else inp_slice.stop
  assert start >= 0
  assert stop <= size
  return slice(start, stop, None)


@dataclasses.dataclass(frozen=True)
class FullMask(Mask):
  """Lazy full mask, allows all tokens to attend to all other tokens."""

  # TODO(amagni): Transform FullMask into a _ComputableMask.

  _shape: tuple[int, int]

  def __post_init__(self):
    if not isinstance(self.shape, tuple):
      raise ValueError(f'Unsupported shape type: {type(self.shape)}')

  @property
  def shape(self) -> Tuple[int, ...]:
    return self._shape

  def __getitem__(self, idx) -> np.ndarray:
    if len(idx) != 2:
      raise NotImplementedError(f'Unsupported slice: {idx}')
    i, j = idx
    if not isinstance(i, slice) or not isinstance(j, slice):
      raise NotImplementedError(f'Unsupported slice: {idx}')
    i = _fill_slice(i, self.shape[0])
    j = _fill_slice(j, self.shape[1])
    return np.ones((i.stop - i.start, j.stop - j.start), dtype=np.bool_)

  def __eq__(self, other: 'FullMask'):
    if not isinstance(other, type(self)):
      return NotImplemented

    return self.shape == other.shape

  def __hash__(self):
    return hash((type(self), self.shape))