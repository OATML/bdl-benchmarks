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

import csv
import io
import os
from typing import Sequence

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

cv2 = tfds.core.lazy_imports.cv2


class DiabeticRetinopathyDiagnosisConfig(tfds.core.BuilderConfig):
  """BuilderConfig for DiabeticRetinopathyDiagnosis."""

  def __init__(
      self,
      target_height: int,
      target_width: int,
      crop: bool = False,
      scale: int = 500,
      **kwargs,
  ):
    """BuilderConfig for DiabeticRetinopathyDiagnosis.

    Args:
      target_height: `int`, number of pixels in height.
      target_width: `int`, number of pixels in width.
      scale: (optional) `int`, the radius of the neighborhood to apply
        Gaussian blur filtering.
      **kwargs: keyword arguments forward to super.
    """
    super(DiabeticRetinopathyDiagnosisConfig, self).__init__(**kwargs)
    self._target_height = target_height
    self._target_width = target_width
    self._scale = scale

  @property
  def target_height(self) -> int:
    return self._target_height

  @property
  def target_width(self) -> int:
    return self._target_width

  @property
  def scale(self) -> int:
    return self._scale


class DiabeticRetinopathyDiagnosis(tfds.image.DiabeticRetinopathyDetection):

  BUILDER_CONFIGS: Sequence[DiabeticRetinopathyDiagnosisConfig] = [
      DiabeticRetinopathyDiagnosisConfig(
          name="medium",
          version="0.0.1",
          description="Images for Medium level.",
          target_height=256,
          target_width=256,
      ),
      DiabeticRetinopathyDiagnosisConfig(
          name="realworld",
          version="0.0.1",
          description="Images for RealWorld level.",
          target_height=512,
          target_width=512,
      ),
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    return tfds.core.DatasetInfo(
        builder=self,
        description="A large set of high-resolution retina images taken under "
        "a variety of imaging conditions. "
        "Ehanced contrast and resized to {}x{}.".format(
            self.builder_config.target_height,
            self.builder_config.target_width),
        features=tfds.features.FeaturesDict({
            "name":
                tfds.features.Text(),  # patient ID + eye. eg: "4_left".
            "image":
                tfds.features.Image(shape=(
                    self.builder_config.target_height,
                    self.builder_config.target_width,
                    3,
                )),
            # 0: (no DR)
            # 1: (with DR)
            "label":
                tfds.features.ClassLabel(num_classes=2),
        }),
        urls=["https://www.kaggle.com/c/diabetic-retinopathy-detection/data"],
        citation=tfds.image.diabetic_retinopathy_detection._CITATION,
    )

  def _generate_examples(self, images_dir_path, csv_path=None, csv_usage=None):
    """Yields Example instances from given CSV. Applies contrast enhancement as
    in https://github.com/btgraham/SparseConvNet/tree/kaggle_Diabetic_Retinopat
    hy_competition. Turns the multiclass (i.e. 5 classes) problem to binary
    classification according to
    https://www.nature.com/articles/s41598-017-17876-z.pdf.

    Args:
      images_dir_path: path to dir in which images are stored.
      csv_path: optional, path to csv file with two columns: name of image and
        label. If not provided, just scan image directory, don't set labels.
      csv_usage: optional, subset of examples from the csv file to use based on
        the "Usage" column from the csv.
    """
    if csv_path:
      with tf.io.gfile.GFile(csv_path) as csv_f:
        reader = csv.DictReader(csv_f)
        data = [(row["image"], int(row["level"]))
                for row in reader
                if csv_usage is None or row["Usage"] == csv_usage]
    else:
      data = [(fname[:-5], -1)
              for fname in tf.io.gfile.listdir(images_dir_path)
              if fname.endswith(".jpeg")]
    for name, label in data:
      record = {
          "name":
              name,
          "image":
              self._preprocess(
                  tf.io.gfile.GFile("%s/%s.jpeg" % (images_dir_path, name),
                                    mode="rb"),
                  target_height=self.builder_config.target_height,
                  target_width=self.builder_config.target_width,
              ),
          "label":
              int(label > 1),
      }

      yield record

  @classmethod
  def _preprocess(
      cls,
      image_fobj,
      target_height: int,
      target_width: int,
      crop: bool = False,
      scale: int = 500,
  ) -> io.BytesIO:
    """Resize an image to have (roughly) the given number of target pixels.

    Args:
      image_fobj: File object containing the original image.
      target_height: `int`, number of pixels in height.
      target_width: `int`, number of pixels in width.
      crops: (optional) `bool`, if True crops the centre of the original
        image t the target size.
      scale: (optional) `int`, the radius of the neighborhood to apply
        Gaussian blur filtering.

    Returns:
      A file object.
    """
    # Decode image using OpenCV2.
    image = cv2.imdecode(np.fromstring(image_fobj.read(), dtype=np.uint8),
                         flags=3)
    try:
      a = cls._get_radius(image, scale)
      b = np.zeros(a.shape)
      cv2.circle(img=b,
                 center=(a.shape[1] // 2, a.shape[0] // 2),
                 radius=int(scale * 0.9),
                 color=(1, 1, 1),
                 thickness=-1,
                 lineType=8,
                 shift=0)
      image = cv2.addWeighted(src1=a,
                              alpha=4,
                              src2=cv2.GaussianBlur(
                                  src=a, ksize=(0, 0), sigmaX=scale // 30),
                              beta=-4,
                              gamma=128) * b + 128 * (1 - b)
    except cv2.error:
      pass
    # Reshape image to target size
    image = cv2.resize(image, (target_height, target_width))
    # Encode the image with quality=72 and store it in a BytesIO object.
    _, buff = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    return io.BytesIO(buff.tostring())

  @staticmethod
  def _get_radius(img: np.ndarray, scale: int) -> np.ndarray:
    """Returns radius of the circle to use.

    Args:
      img: `numpy.ndarray`, an image, with shape [height, width, 3].
      scale: `int`, the radius of the neighborhood.

    Returns:
      A resized image.
    """
    x = img[img.shape[0] // 2, ...].sum(axis=1)
    r = 0.5 * (x > x.mean() // 10).sum()
    s = scale * 1.0 / r
    return cv2.resize(img, (0, 0), fx=s, fy=s)
