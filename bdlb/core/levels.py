# Copyright 2019 BDL Benchmarks Authors. All Rights Reserved.
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
"""Downstream tasks levels."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum
from typing import Text


class Level(enum.IntEnum):
  """Downstream task levels.

  TOY: Fewer examples and drastically reduced input dimensionality.
    This version is intended for sanity checks and debugging only.
    Training on a modern CPU should take five to ten minutes.
  MEDIUM: Fewer examples and reduced input dimensionality.
    This version is intended for prototyping before moving on to the
    real-world scale data. Training on a single modern GPU should take
    five or six hours.
  REALWORLD: The full dataset and input dimensionality. This version is
    intended for the evaluation of proposed techniques at a scale applicable
    to the real world. There are no guidelines for train time for the real-world
    version of the task, reflecting the fact that any improvement will translate to
    safer, more robust and reliable systems.
  """

  TOY = 0
  MEDIUM = 1
  REALWORLD = 2

  @classmethod
  def from_str(cls, strvalue: Text) -> "Level":
    """Parses a string value to ``Level``.

    Args:
      strvalue: `str`, the level in string format.

    Returns:
      The `IntEnum` ``Level`` object.
    """
    strvalue = strvalue.lower()
    if strvalue == "toy":
      return cls.TOY
    elif strvalue == "medium":
      return cls.MEDIUM
    elif strvalue == "realworld":
      return cls.REALWORLD
    else:
      raise ValueError(
          "Unrecognized level value '{}' provided.".format(strvalue))
