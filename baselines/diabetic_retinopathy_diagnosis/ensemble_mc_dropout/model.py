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
"""Uncertainty estimator for the Ensemble Monte Carlo Dropout baseline."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def predict(x, models, num_samples, type="entropy"):
  """Deep Ensembles uncertainty estimator.

  Args:
    x: `numpy.ndarray`, datapoints from input space,
      with shape [B, H, W, 3], where B the batch size and
      H, W the input images height and width accordingly.
    num_samples: `int`, number of Monte Carlo samples
      (i.e. forward passes from dropout) used for
      the calculation of predictive mean and uncertainty.
    type: (optional) `str`, type of uncertainty returns,
      one of {"entropy", "stddev"}.
    models: `iterable` of `tensorflow.keras.Model`,
      a probabilistic model, which accepts input with shape
      [B, H, W, 3] and outputs sigmoid probability [0.0, 1.0],
      and also accepts boolean arguments `training=True` for
      enabling dropout at test time.

  Returns:
    mean: `numpy.ndarray`, predictive mean, with shape [B].
    uncertainty: `numpy.ndarray`, ncertainty in prediction,
      with shape [B].
  """
  import numpy as np
  import scipy.stats

  # Get shapes of data
  B, _, _, _ = x.shape

  # Monte Carlo samples from different dropout mask at test time from different models
  mc_samples = np.asarray([
      model(x, training=True) for _ in range(num_samples) for model in models
  ]).reshape(-1, B)

  # Bernoulli output distribution
  dist = scipy.stats.bernoulli(mc_samples.mean(axis=0))

  # Predictive mean calculation
  mean = dist.mean()

  # Use predictive entropy for uncertainty
  if type == "entropy":
    uncertainty = dist.entropy()
  # Use predictive standard deviation for uncertainty
  elif type == "stddev":
    uncertainty = dist.std()
  else:
    raise ValueError(
        "Unrecognized type={} provided, use one of {'entropy', 'stddev'}".
        format(type))

  return mean, uncertainty
