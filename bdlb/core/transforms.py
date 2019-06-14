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
"""Data augmentation and transformations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Transform(object):
  """Abstract transformation class."""

  def __call__(self):
    raise NotImplementedError()


class Compose(Transform):
  """Uber transformation, composing a list of transformations."""

  def __init__(self, trans):
    """Constructs a composition of transformations.
    
    Args:
      trans: `iterable`, sequence of transformations to be composed.
    """
    self.trans = trans

  def __call__(self, x):
    """Returns a composite function of transformations.
    
    Args:
      x: `any`, raw data format.
    
    Returns:
      A composite function to be used with `tf.data.Dataset.map()`.
    """
    import inspect
    for f in self.trans:
      nargs = len(inspect.signature(f).parameters)
      if "y" in locals():
        x, y = f(x, y)
      else:
        if nargs == 1:
          x = f(x)
        else:
          x, y = f(x[0], x[1])
    if "y" in locals():
      return x, y
    else:
      return x


class RandomAugment(Transform):

  def __init__(self, **config):
    """Constructs a tranformer from `config`.
    
    Args:
      **config: keyword arguments for
        `tensorflow.keras.preprocessing.image.ImageDataGenerator`
    """
    self.idg = tf.keras.preprocessing.image.ImageDataGenerator(**config)

  def __call__(self, x, y):
    """Returns a randomly augmented image and its label.
    
    Args:
      x: `tensorflow.Tensor`, an image, with shape [height, width, channels].
      y: `tensorflow.Tensor`, a target, with shape [].
    
    
    Returns:
      The processed tuple:
        * `x`: `tensorflow.Tensor`, the randomly augmented image,
          with shape [height, width, channels].
        * `y`: `tensorflow.Tensor`, the unchanged target, with shape [].
    """
    return tf.py_function(self._transform, inp=[x], Tout=tf.float32), y

  def _transform(self, x):
    """Helper function for `tensorflow.py_function`, will
    be removed when TensorFlow 2.0 is released."""
    return tf.cast(self.idg.random_transform(x.numpy()), tf.float32)


class Resize(Transform):

  def __init__(self, target_height, target_width):
    """Constructs an image resizer.
    
    Args:
      target_height: `int`, number of pixels in height.
      target_width: `int`, number of pixels in width.
    """
    self.target_height = target_height
    self.target_width = target_width

  def __call__(self, x, y):
    """Returns a resized image."""
    return tf.image.resize(x, size=[self.target_height, self.target_width]), y


class Normalize(Transform):

  def __init__(self, loc, scale):
    self.loc = loc
    self.scale = scale

  def __call__(self, x, y):
    return (x - self.loc) / self.scale, y