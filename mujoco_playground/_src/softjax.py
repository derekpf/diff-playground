# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Project-wide softjax configuration."""

import functools
import numbers
from typing import Final

import softjax as _softjax

_DEFAULT_SOFTNESS: Final[float] = 0.0


def _with_softness(fn, softness_position):
  """Adds the project default and maps zero softness to hard mode."""

  mode_position = softness_position + 1

  @functools.wraps(fn)
  def wrapped(*args, **kwargs):
    if len(args) <= softness_position and "softness" not in kwargs:
      kwargs["softness"] = _DEFAULT_SOFTNESS
    softness = (
        args[softness_position]
        if len(args) > softness_position
        else kwargs.get("softness")
    )
    if (
        isinstance(softness, numbers.Real)
        and softness == 0.0
        and len(args) <= mode_position
        and "mode" not in kwargs
    ):
      kwargs["mode"] = "hard"
    return fn(*args, **kwargs)

  return wrapped


def _with_st_enable(st_fn, soft_fn, softness_position):
  """Selects straight-through or ordinary behavior for an adapter."""

  wrapped_st = _with_softness(st_fn, softness_position)
  wrapped_soft = _with_softness(soft_fn, softness_position)

  @functools.wraps(st_fn)
  def wrapped(*args, st_enable: bool = True, **kwargs):
    fn = wrapped_st if st_enable else wrapped_soft
    return fn(*args, **kwargs)

  return wrapped


# Keep the external softjax argument order while providing the project default
# when callers omit softness.
abs = _with_softness(_softjax.abs, 1)
clip = _with_softness(_softjax.clip, 3)
greater = _with_softness(_softjax.greater, 2)
greater_st = _with_st_enable(_softjax.greater_st, greater, 2)
greater_equal = _with_softness(_softjax.greater_equal, 2)
greater_equal_st = _with_st_enable(_softjax.greater_equal_st, greater_equal, 2)
less = _with_softness(_softjax.less, 2)
less_st = _with_st_enable(_softjax.less_st, less, 2)
less_equal = _with_softness(_softjax.less_equal, 2)
less_equal_st = _with_st_enable(_softjax.less_equal_st, less_equal, 2)
max = _with_softness(_softjax.max, 3)
max_st = _with_st_enable(_softjax.st(_softjax.max), max, 3)
min = _with_softness(_softjax.min, 3)
min_st = _with_st_enable(_softjax.st(_softjax.min), min, 3)
relu = _with_softness(_softjax.relu, 1)
relu_st = _with_st_enable(_softjax.st(_softjax.relu), relu, 1)
abs_st = _with_st_enable(_softjax.st(_softjax.abs), abs, 1)
clip_st = _with_st_enable(_softjax.st(_softjax.clip), clip, 3)

# Operations without a softness argument are passed through unchanged.
all = _softjax.all
any = _softjax.any
arccos = _softjax.arccos
arcsin = _softjax.arcsin
div = _softjax.div
logical_and = _softjax.logical_and
logical_not = _softjax.logical_not
logical_or = _softjax.logical_or
norm = _softjax.norm
where = _softjax.where
