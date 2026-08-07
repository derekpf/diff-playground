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
from typing import Final

import softjax as _softjax

_DEFAULT_SOFTNESS: Final[float] = 0.01


def _with_softness(fn, softness_position):
  """Adds the project default when a softness value is not provided."""

  @functools.wraps(fn)
  def wrapped(*args, **kwargs):
    if len(args) <= softness_position and "softness" not in kwargs:
      kwargs["softness"] = _DEFAULT_SOFTNESS
    return fn(*args, **kwargs)

  return wrapped


# Keep the external softjax argument order while providing the project default
# when callers omit softness.
abs = _with_softness(_softjax.abs, 1)
clip = _with_softness(_softjax.clip, 3)
greater = _with_softness(_softjax.greater, 2)
greater_st = _with_softness(_softjax.greater_st, 2)
greater_equal = _with_softness(_softjax.greater_equal, 2)
greater_equal_st = _with_softness(_softjax.greater_equal_st, 2)
less = _with_softness(_softjax.less, 2)
less_st = _with_softness(_softjax.less_st, 2)
less_equal = _with_softness(_softjax.less_equal, 2)
less_equal_st = _with_softness(_softjax.less_equal_st, 2)
max = _with_softness(_softjax.max, 3)
min = _with_softness(_softjax.min, 3)
relu = _with_softness(_softjax.relu, 1)

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
