# Copyright 2018 BDL Benchmarks Authors. All Rights Reserved.
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
"""Data structures and API for general benchmarks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from .levels import Level

# Abstract class for benchmark information.
BenchmarkInfo = collections.namedtuple("BenchmarkInfo", [
    "description",
    "urls",
    "setup",
    "citation",
])

# Container for train, validation and test sets.
DataSplits = collections.namedtuple("DataSplits", [
    "train",
    "validation",
    "test",
])


class Benchmark(object):
  """Abstract class for benchmark objects, specifying the core API."""

  def download_and_prepare(self):
    """Downloads and prepares necessary datasets for benchmark."""
    raise NotImplementedError()

  @property
  def info(self) -> BenchmarkInfo:
    """Text description of the benchmark."""
    raise NotImplementedError()

  @property
  def level(self) -> Level:
    """The downstream task level."""
    raise NotImplementedError()
