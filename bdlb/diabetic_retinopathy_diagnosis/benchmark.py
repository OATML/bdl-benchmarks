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
"""Diabetic retinopathy diagnosis BDL Benchmark."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Text
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf
from absl import logging

from ..core import transforms
from ..core.benchmark import Benchmark
from ..core.benchmark import BenchmarkInfo
from ..core.benchmark import DataSplits
from ..core.constants import DATA_DIR
from ..core.levels import Level

tfk = tf.keras

_DIABETIC_RETINOPATHY_DIAGNOSIS_DATA_DIR = os.path.join(
    DATA_DIR, "downloads", "manual", "diabetic_retinopathy_diagnosis")


class DiabeticRetinopathyDiagnosisBecnhmark(Benchmark):
  """Diabetic retinopathy diagnosis benchmark class."""

  def __init__(
      self,
      level: Union[Text, Level],
      batch_size: int = 64,
      data_dir: Optional[Text] = None,
      download_and_prepare: bool = False,
  ):
    """Constructs a benchmark object.

    Args:
      level: `Level` or `str, downstream task level.
      batch_size: (optional) `int`, number of datapoints
        per mini-batch.
      data_dir: (optional) `str`, path to parent data directory.
      download_and_prepare: (optional) `bool`, if the data is not available
        it downloads and preprocesses it.
    """
    self.__level = level if isinstance(level, Level) else Level.from_str(level)
    try:
      self.__ds = self.load(level=level,
                            batch_size=batch_size,
                            data_dir=data_dir or DATA_DIR)
    except AssertionError:
      if not download_and_prepare:
        raise
      else:
        logging.info(
            "Data not found, `DiabeticRetinopathyDiagnosisBecnhmark.download_and_prepare()`"
            " is now running...")
        self.download_and_prepare()

  @classmethod
  def evaluate(
      cls,
      estimator: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
      dataset: tf.data.Dataset,
      output_dir: Optional[Text] = None,
      name: Optional[Text] = None,
  ) -> Dict[Text, float]:
    """Evaluates an `estimator` on the `mode` benchmark dataset.

    Args:
      estimator: `lambda x: mu_x, uncertainty_x`, an uncertainty estimation
        function, which returns `mean_x` and predictive `uncertainty_x`.
      dataset: `tf.data.Dataset`, on which dataset to performance evaluation.
      output_dir: (optional) `str`, directory to save figures.
      name: (optional) `str`, the name of the method.
    """
    import inspect
    import tqdm
    import tensorflow_datasets as tfds
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt

    # Containers used for caching performance evaluation
    y_true = list()
    y_pred = list()
    y_uncertainty = list()

    # Convert to NumPy iterator if necessary
    ds = dataset if inspect.isgenerator(dataset) else tfds.as_numpy(dataset)

    for x, y in tqdm.tqdm(ds):
      # Sample from probabilistic model
      mean, uncertainty = estimator(x)
      # Cache predictions
      y_true.append(y)
      y_pred.append(mean)
      y_uncertainty.append(uncertainty)

    # Use vectorized NumPy containers
    y_true = np.concatenate(y_true).flatten()
    y_pred = np.concatenate(y_pred).flatten()
    y_uncertainty = np.concatenate(y_uncertainty).flatten()
    fractions = np.asarray([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # Metrics for evaluation
    metrics = zip(["accuracy", "auc"], cls.metrics())

    return {
        metric: cls._evaluate_metric(
            y_true,
            y_pred,
            y_uncertainty,
            fractions,
            lambda y_true, y_pred: metric_fn(y_true, y_pred).numpy(),
            name,
        ) for (metric, metric_fn) in metrics
    }

  @staticmethod
  def _evaluate_metric(
      y_true: np.ndarray,
      y_pred: np.ndarray,
      y_uncertainty: np.ndarray,
      fractions: Sequence[float],
      metric_fn: Callable[[np.ndarray, np.ndarray], float],
      name=None,
  ) -> pd.DataFrame:
    """Evaluate model predictive distribution on `metric_fn` at data retain
    `fractions`.

    Args:
      y_true: `numpy.ndarray`, the ground truth labels, with shape [N].
      y_pred: `numpy.ndarray`, the model predictions, with shape [N].
      y_uncertainty: `numpy.ndarray`, the model uncertainties,
        with shape [N].
      fractions: `iterable`, the percentages of data to retain for
        calculating `metric_fn`.
      metric_fn: `lambda(y_true, y_pred) -> float`, a metric
        function that provides a score given ground truths
        and predictions.
      name: (optional) `str`, the name of the method.

    Returns:
      A `pandas.DataFrame` with columns ["retained_data", "mean", "std"],
      that summarizes the scores at different data retained fractions.
    """

    N = y_true.shape[0]

    # Sorts indexes by ascending uncertainty
    I_uncertainties = np.argsort(y_uncertainty)

    # Score containers
    mean = np.empty_like(fractions)
    # TODO(filangel): do bootstrap sampling and estimate standard error
    std = np.zeros_like(fractions)

    for i, frac in enumerate(fractions):
      # Keep only the %-frac of lowest uncertainties
      I = np.zeros(N, dtype=bool)
      I[I_uncertainties[:int(N * frac)]] = True
      mean[i] = metric_fn(y_true[I], y_pred[I])

    # Store
    df = pd.DataFrame(dict(retained_data=fractions, mean=mean, std=std))
    df.name = name

    return df

  @property
  def datasets(self) -> tf.data.Dataset:
    """Pointer to the processed datasets."""
    return self.__ds

  @property
  def info(self) -> BenchmarkInfo:
    """Text description of the benchmark."""
    return BenchmarkInfo(description="", urls="", setup="", citation="")

  @property
  def level(self) -> Level:
    """The downstream task level."""
    return self.__level

  @staticmethod
  def loss() -> tfk.losses.Loss:
    """Loss used for training binary classifiers."""
    return tfk.losses.BinaryCrossentropy()

  @staticmethod
  def metrics() -> tfk.metrics.Metric:
    """Evaluation metrics used for monitoring training."""
    return [tfk.metrics.BinaryAccuracy(), tfk.metrics.AUC()]

  @staticmethod
  def class_weight() -> Sequence[float]:
    """Class weights used for rebalancing the dataset, by skewing the `loss`
    accordingly."""
    return [1.0, 4.0]

  @classmethod
  def load(
      cls,
      level: Union[Text, Level] = "realworld",
      batch_size: int = 64,
      data_dir: Optional[Text] = None,
      as_numpy: bool = False,
  ) -> DataSplits:
    """Loads the datasets for the benchmark.

    Args:
      level: `Level` or `str, downstream task level.
      batch_size: (optional) `int`, number of datapoints
        per mini-batch.
      data_dir: (optional) `str`, path to parent data directory.
      as_numpy: (optional) `bool`, if True returns python generators
        with `numpy.ndarray` outputs.

    Returns:
      A namedtuple with properties:
        * train: `tf.data.Dataset`, train dataset.
        * validation: `tf.data.Dataset`, validation dataset.
        * test: `tf.data.Dataset`, test dataset.
    """
    import tensorflow_datasets as tfds
    from .tfds_adapter import DiabeticRetinopathyDiagnosis

    # Fetch datasets
    try:
      ds_train, ds_validation, ds_test = DiabeticRetinopathyDiagnosis(
          data_dir=data_dir or DATA_DIR,
          config=level).as_dataset(split=["train", "validation", "test"],
                                   shuffle_files=True,
                                   batch_size=batch_size)
    except AssertionError as ae:
      raise AssertionError(
          str(ae) +
          " Run DiabeticRetinopathyDiagnosisBecnhmark.download_and_prepare()"
          " first and then retry.")

    # Parse task level
    level = level if isinstance(level, Level) else Level.from_str(level)
    # Dataset tranformations
    transforms_train, transforms_eval = cls._preprocessors()
    # Apply transformations
    ds_train = ds_train.map(transforms_train,
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_validation = ds_validation.map(
        transforms_eval, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.map(transforms_eval,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Prefetches datasets to memory
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    ds_validation = ds_validation.prefetch(tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    if as_numpy:
      # Convert to NumPy iterators
      ds_train = tfds.as_numpy(ds_train)
      ds_validation = tfds.as_numpy(ds_validation)
      ds_test = tfds.as_numpy(ds_test)

    return DataSplits(ds_train, ds_validation, ds_test)

  @classmethod
  def download_and_prepare(cls, levels=None) -> None:
    """Downloads dataset from Kaggle, extracts zip files and processes it using
    `tensorflow_datasets`.

    Args:
      levels: (optional) `iterable` of `str`, specifies which
        levels from {'medium', 'realworld'} to prepare,
        if None it prepares all the levels.

    Raises:
      OSError: if `~/.kaggle/kaggle.json` is not set up.
    """
    # Disable GPU for data download, extraction and preparation
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    cls._download()
    # cls._extract()
    #cls._prepare(levels)

  @staticmethod
  def _download() -> None:
    """Downloads data from Kaggle using `tensorflow_datasets`.

    Raises:
      OSError: if `~/.kaggle/kaggle.json` is not set up.
    """
    import subprocess as sp
    import tensorflow_datasets as tfds

    # Append `/home/$USER/.local/bin` to path
    os.environ["PATH"] += ":/home/{}/.local/bin/".format(os.environ["USER"])

    # Download all files from Kaggle
    drd = tfds.download.kaggle.KaggleCompetitionDownloader(
        "diabetic-retinopathy-detection")
    try:
      for dfile in drd.competition_files:
        drd.download_file(dfile,
                          output_dir=_DIABETIC_RETINOPATHY_DIAGNOSIS_DATA_DIR)
    except sp.CalledProcessError as cpe:
      raise OSError(
          str(cpe) + "." +
          " Make sure you have ~/.kaggle/kaggle.json setup, fetched from the Kaggle website"
          " https://www.kaggle.com/<username>/account -> 'Create New API Key'."
          " Also accept the dataset license by going to"
          " https://www.kaggle.com/c/diabetic-retinopathy-detection/rules"
          " and look for the button 'I Understand and Accept' (make sure when reloading the"
          " page that the button does not pop up again).")

  @staticmethod
  def _extract() -> None:
    """Extracts zip files downloaded from Kaggle."""
    import glob
    import tqdm
    import zipfile
    import tempfile

    # Extract train and test original images
    for split in ["train", "test"]:
      # Extract "<split>.zip.00*"" files to "<split>"
      with tempfile.NamedTemporaryFile() as tmp:
        # Concatenate "<split>.zip.00*" to "<split>.zip"
        for fname in tqdm.tqdm(
            sorted(
                glob.glob(
                    os.path.join(_DIABETIC_RETINOPATHY_DIAGNOSIS_DATA_DIR,
                                 "{split}.zip.00*".format(split=split))))):
          # Unzip "<split>.zip" to "<split>"
          with open(fname, "rb") as ztmp:
            tmp.write(ztmp.read())
        with zipfile.ZipFile(tmp) as zfile:
          for image in tqdm.tqdm(iterable=zfile.namelist(),
                                 total=len(zfile.namelist())):
            zfile.extract(member=image,
                          path=_DIABETIC_RETINOPATHY_DIAGNOSIS_DATA_DIR)
      # Delete "<split>.zip.00*" files
      for splitzip in os.listdir(_DIABETIC_RETINOPATHY_DIAGNOSIS_DATA_DIR):
        if "{split}.zip.00".format(split=split) in splitzip:
          os.remove(
              os.path.join(_DIABETIC_RETINOPATHY_DIAGNOSIS_DATA_DIR, splitzip))

    # Extract "sample.zip", "trainLabels.csv.zip"
    for fname in ["sample", "trainLabels.csv"]:
      zfname = os.path.join(_DIABETIC_RETINOPATHY_DIAGNOSIS_DATA_DIR,
                            "{fname}.zip".format(fname=fname))
      with zipfile.ZipFile(zfname) as zfile:
        zfile.extractall(_DIABETIC_RETINOPATHY_DIAGNOSIS_DATA_DIR)
      os.remove(zfname)

  @staticmethod
  def _prepare(levels=None) -> None:
    """Generates the TFRecord objects for medium and realworld experiments."""
    import multiprocessing
    from absl import logging
    from .tfds_adapter import DiabeticRetinopathyDiagnosis
    # Hangle each level individually
    for level in levels or ["medium", "realworld"]:
      dtask = DiabeticRetinopathyDiagnosis(data_dir=DATA_DIR, config=level)
      logging.debug("=== Preparing TFRecords for {} ===".format(level))
      dtask.download_and_prepare()

  @classmethod
  def _preprocessors(cls) -> Tuple[transforms.Transform, transforms.Transform]:
    """Applies transformations to the raw data."""
    import tensorflow_datasets as tfds

    # Transformation hyperparameters
    mean = np.asarray([0.42606387, 0.29752496, 0.21309826])
    stddev = np.asarray([0.27662534, 0.20280295, 0.1687619])

    class Parse(transforms.Transform):
      """Parses datapoints from raw `tf.data.Dataset`."""

      def __call__(self, x, y=None):
        """Returns `as_supervised` tuple."""
        return x["image"], x["label"]

    class CastX(transforms.Transform):
      """Casts image to `dtype`."""

      def __init__(self, dtype):
        """Constructs a type caster."""
        self.dtype = dtype

      def __call__(self, x, y):
        """Returns casted image (to `dtype`) and its (unchanged) label as
        tuple."""
        return tf.cast(x, self.dtype), y

    class To01X(transforms.Transform):
      """Rescales image to [min, max]=[0, 1]."""

      def __call__(self, x, y):
        """Returns rescaled image and its (unchanged) label as tuple."""
        return x / 255.0, y

    # Get augmentation schemes
    [augmentation_config,
     no_augmentation_config] = cls._ImageDataGenerator_config()

    # Transformations for train dataset
    transforms_train = transforms.Compose([
        Parse(),
        CastX(tf.float32),
        To01X(),
        transforms.Normalize(mean, stddev),
        # TODO(filangel): hangle batch with ImageDataGenerator
        # transforms.RandomAugment(**augmentation_config),
    ])

    # Transformations for validation/test dataset
    transforms_eval = transforms.Compose([
        Parse(),
        CastX(tf.float32),
        To01X(),
        transforms.Normalize(mean, stddev),
        # TODO(filangel): hangle batch with ImageDataGenerator
        # transforms.RandomAugment(**no_augmentation_config),
    ])

    return transforms_train, transforms_eval

  @staticmethod
  def _ImageDataGenerator_config():
    """Returns the configs for the
    `tensorflow.keras.preprocessing.image.ImageDataGenerator`, used for the
    random augmentation of the dataset, following the implementation of
    https://github.com/chleibig/disease-detection/blob/f3401b26aa9b832ff77afe93
    e3faa342f7d088e5/scripts/inspect_data_augmentation.py."""
    augmentation_config = dict(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=180.0,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.,
        zoom_range=0.10,
        channel_shift_range=0.,
        fill_mode="constant",
        cval=0.,
        horizontal_flip=True,
        vertical_flip=True,
        data_format="channels_last",
    )
    no_augmentation_config = dict(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0.0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        shear_range=0.,
        zoom_range=0.0,
        channel_shift_range=0.,
        fill_mode="nearest",
        cval=0.,
        horizontal_flip=False,
        vertical_flip=False,
        data_format="channels_last",
    )
    return augmentation_config, no_augmentation_config
