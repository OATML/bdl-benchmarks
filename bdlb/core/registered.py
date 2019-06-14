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

from ..diabetic_retinopathy_diagnosis.benchmark import DiabeticRetinopathyDiagnosisBecnhmark

# Internal registry containing <str registered_name, Benchmark subclass>
_BENCHMARK_REGISTRY = {
    "diabetic_retinopathy_diagnosis": DiabeticRetinopathyDiagnosisBecnhmark
}


def load(benchmark,
         level=None,
         data_dir=None,
         download_and_prepare=True,
         **dtask_kwargs):
  """Loads the named benchmark into a `bdlb.Benchmark`.
  
  Args:
    benchmark: `str`, the registerd name of `bdlb.Benchmark`.
    level: `bdlb.Level` or `str` (optional), which level of the
      benchmark to load. If None, will return the realworld level.
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
      **dtask_kwargs)


class BenchmarkNotFoundError(ValueError):
  """The requested `bdlb.Benchmark` was not found."""

  def __init__(self, name):
    all_denchmarks_str = "\n\t- ".join([""] + list(_BENCHMARK_REGISTRY.keys()))
    error_str = (
        "Benchmark {name} not found. Available denchmarks: {benchmarks}\n",
        format(name=name, benchmarks=all_denchmarks_str))
    super(BenchmarkNotFoundError, self).__init__(error_str)