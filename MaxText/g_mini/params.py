"""Utils for loading G_mini params.

These utilities are just helpers for current development. They will not be
needed once G_mini switches to Orbax and changes checkpoint formats ahead of
open sourcing.

TODO(g_mini): Convert G_mini checkpoints to format native to model structure.
"""

import functools
from typing import Any

import orbax


@functools.cache
def load_params(path: str) -> dict[str, Any]:
  """Loads parameters from a checkpoint path."""
  checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  params = checkpointer.restore(path)
  return params


def param_remapper(orig_params: dict[str, Any]) -> dict[str, Any]:
  """Remaps params to new module layout.

  This is needed here because the model definition  does not have a separate
  `mlp` module. For the real code release, we will just save the params in a
  different format and this will not be needed.

  Args:
    orig_params: original dict of parameters in G_mini format.

  Returns:
    dict of params with different names.
  """
  new_params = {}
  for k, v in orig_params.items():
    if 'mlp/' in k:
      layer_name, param = k.rsplit('/', maxsplit=1)
      if layer_name not in new_params:
        new_params[layer_name] = {}
      if 'w' in v:
        new_params[layer_name][param] = v['w']
    else:
      new_params[k] = v
  return new_params


def nest_params(params: dict[str, Any]) -> dict[str, Any]:
  """Nests params as a dict of dicts rather than a flat dict."""
  nested_params = {}
  for path, param in params.items():
    *path, leaf = path.split('/')
    subdict = nested_params
    for key in path:
      subdict = subdict.setdefault(key, {})
    subdict[leaf] = param
  return nested_params