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
"""Model definition of the VGGish network for Mean-Field Variational Inference
baseline."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools


def VGGFlipout(num_base_filters, learning_rate, input_shape):
  """VGG-like model with Flipout for diabetic retinopathy diagnosis.

  Args:
    num_base_filters: `int`, number of convolution filters in the
      first layer.
    learning_rate: `float`, ADAM optimizer learning rate.
    input_shape: `iterable`, the shape of the images in the input layer.

  Returns:
    A tensorflow.keras.Sequential VGG-like model with flipout.
  """
  import tensorflow as tf
  tfk = tf.keras
  tfkl = tfk.layers
  import tensorflow_probability as tfp
  tfpl = tfp.layers
  from bdlb.diabetic_retinopathy_diagnosis.benchmark import DiabeticRetinopathyDiagnosisBecnhmark

  # Feedforward neural network
  model = tfk.Sequential([
      tfkl.InputLayer(input_shape),
      # Block 1
      tfpl.Convolution2DFlipout(filters=num_base_filters,
                                kernel_size=3,
                                strides=(2, 2),
                                padding="same"),
      tfkl.Activation("relu"),
      tfkl.MaxPooling2D(pool_size=3, strides=(2, 2), padding="same"),
      # Block 2
      tfpl.Convolution2DFlipout(filters=num_base_filters,
                                kernel_size=3,
                                strides=(1, 1),
                                padding="same"),
      tfkl.Activation("relu"),
      tfpl.Convolution2DFlipout(filters=num_base_filters,
                                kernel_size=3,
                                strides=(1, 1),
                                padding="same"),
      tfkl.Activation("relu"),
      tfkl.MaxPooling2D(pool_size=3, strides=(2, 2), padding="same"),
      # Block 3
      tfpl.Convolution2DFlipout(filters=num_base_filters * 2,
                                kernel_size=3,
                                strides=(1, 1),
                                padding="same"),
      tfkl.Activation("relu"),
      tfpl.Convolution2DFlipout(filters=num_base_filters * 2,
                                kernel_size=3,
                                strides=(1, 1),
                                padding="same"),
      tfkl.Activation("relu"),
      tfkl.MaxPooling2D(pool_size=3, strides=(2, 2), padding="same"),
      # Block 4
      tfpl.Convolution2DFlipout(filters=num_base_filters * 4,
                                kernel_size=3,
                                strides=(1, 1),
                                padding="same"),
      tfkl.Activation("relu"),
      tfpl.Convolution2DFlipout(filters=num_base_filters * 4,
                                kernel_size=3,
                                strides=(1, 1),
                                padding="same"),
      tfkl.Activation("relu"),
      tfpl.Convolution2DFlipout(filters=num_base_filters * 4,
                                kernel_size=3,
                                strides=(1, 1),
                                padding="same"),
      tfkl.Activation("relu"),
      tfpl.Convolution2DFlipout(filters=num_base_filters * 4,
                                kernel_size=3,
                                strides=(1, 1),
                                padding="same"),
      tfkl.Activation("relu"),
      tfkl.MaxPooling2D(pool_size=3, strides=(2, 2), padding="same"),
      # Block 5
      tfpl.Convolution2DFlipout(filters=num_base_filters * 8,
                                kernel_size=3,
                                strides=(1, 1),
                                padding="same"),
      tfkl.Activation("relu"),
      tfpl.Convolution2DFlipout(filters=num_base_filters * 8,
                                kernel_size=3,
                                strides=(1, 1),
                                padding="same"),
      tfkl.Activation("relu"),
      tfpl.Convolution2DFlipout(filters=num_base_filters * 8,
                                kernel_size=3,
                                strides=(1, 1),
                                padding="same"),
      tfkl.Activation("relu"),
      tfpl.Convolution2DFlipout(filters=num_base_filters * 8,
                                kernel_size=3,
                                strides=(1, 1),
                                padding="same"),
      tfkl.Activation("relu"),
      # Global poolings
      tfkl.Lambda(lambda x: tfk.backend.concatenate(
          [tfkl.GlobalAvgPool2D()
           (x), tfkl.GlobalMaxPool2D()(x)], axis=1)),
      # Fully-connected
      tfpl.DenseFlipout(1,),
      tfkl.Activation("sigmoid")
  ])

  model.compile(loss=DiabeticRetinopathyDiagnosisBecnhmark.loss(),
                optimizer=tfk.optimizers.Adam(learning_rate),
                metrics=DiabeticRetinopathyDiagnosisBecnhmark.metrics())

  return model


def predict(x, model, num_samples, type="entropy"):
  """Monte Carlo dropout uncertainty estimator.

  Args:
    x: `numpy.ndarray`, datapoints from input space,
      with shape [B, H, W, 3], where B the batch size and
      H, W the input images height and width accordingly.
    model: `tensorflow.keras.Model`, a probabilistic model,
      which accepts input with shape [B, H, W, 3] and
      outputs sigmoid probability [0.0, 1.0].
    num_samples: `int`, number of Monte Carlo samples
      (i.e. forward passes from dropout) used for
      the calculation of predictive mean and uncertainty.
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

  # Monte Carlo samples from different dropout mask at test time
  mc_samples = np.asarray([model(x) for _ in range(num_samples)]).reshape(-1, B)

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
