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
"""Benchmarks registry handlers and definitions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict
from typing import Optional
from typing import Text
from typing import Union

from ..core.benchmark import Benchmark
from ..core.levels import Level
from ..diabetic_retinopathy_diagnosis.benchmark import \
    DiabeticRetinopathyDiagnosisBecnhmark

# Internal registry containing <str registered_name, Benchmark subclass>
_BENCHMARK_REGISTRY: Dict[Text, Benchmark] = {
    "diabetic_retinopathy_diagnosis": DiabeticRetinopathyDiagnosisBecnhmark
}


def load(
    benchmark: Text,
    level: Union[Text, Level] = "realworld",
    data_dir: Optional[Text] = None,
    download_and_prepare: bool = True,
    **dtask_kwargs,
) -> Benchmark:
  """Loads the named benchmark into a `bdlb.Benchmark`.

  Args:
    benchmark: `str`, the registerd name of `bdlb.Benchmark`.
    level: `bdlb.Level` or `str`, which level of the benchmark to load.
    data_dir: `str` (optional), directory to read/write data.
      Defaults to "~/.bdlb/data".
    download_and_prepare: (optional) `bool`, if the data is not available
        it downloads and preprocesses it.
    dtask_kwargs: key arguments for the benchmark contructor.

  Returns:
    A registered `bdlb.Benchmark` with `level` at `data_dir`.

  Raises:
    BenchmarkNotFoundError: if `name` is unrecognised.
  """
  if not benchmark in _BENCHMARK_REGISTRY:
    raise BenchmarkNotFoundError(benchmark)
  # Fetch benchmark object
  return _BENCHMARK_REGISTRY.get(benchmark)(
      level=level,
      data_dir=data_dir,
      download_and_prepare=download_and_prepare,
      **dtask_kwargs,
  )


class BenchmarkNotFoundError(ValueError):
  """The requested `bdlb.Benchmark` was not found."""

  def __init__(self, name: Text):
    all_denchmarks_str = "\n\t- ".join([""] + list(_BENCHMARK_REGISTRY.keys()))
    error_str = (
        "Benchmark {name} not found. Available denchmarks: {benchmarks}\n",
        format(name=name, benchmarks=all_denchmarks_str))
    super(BenchmarkNotFoundError, self).__init__(error_str)
