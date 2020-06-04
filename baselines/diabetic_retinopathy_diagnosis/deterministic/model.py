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
"""Uncertainty estimator for the deterministic deep model baseline."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def predict(x, model, num_samples, type="entropy"):
  """Simple sigmoid uncertainty estimator.

  Args:
    x: `numpy.ndarray`, datapoints from input space,
      with shape [B, H, W, 3], where B the batch size and
      H, W the input images height and width accordingly.
    model: `tensorflow.keras.Model`, a probabilistic model,
      which accepts input with shape [B, H, W, 3] and
      outputs sigmoid probability [0.0, 1.0], and also
      accepts boolean arguments `training=False` for
      disabling dropout at test time.
    type: (optional) `str`, type of uncertainty returns,
      one of {"entropy", "stddev"}.

  Returns:
    mean: `numpy.ndarray`, predictive mean, with shape [B].
    uncertainty: `numpy.ndarray`, ncertainty in prediction,
      with shape [B].
  """
  import numpy as np
  import scipy.stats

  # Get shapes of data
  B, _, _, _ = x.shape

  # Single forward pass from the deterministic model
  p = model(x, training=False)

  # Bernoulli output distribution
  dist = scipy.stats.bernoulli(p)

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
