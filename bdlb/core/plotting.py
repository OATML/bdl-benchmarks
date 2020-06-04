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
"""Helper functions for visualization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Dict
from typing import Optional
from typing import Text

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

tfk = tf.keras


def tfk_history(
    history: tfk.callbacks.History,
    output_dir: Optional[Text] = None,
    **ax_set_kwargs,
):
  """Visualization of `tensorflow.keras.callbacks.History`, similar to
  `TensorBoard`, in train and validation.

  Args:
    history: `tensorflow.keras.callbacks.History`, the logs of
      training a `tensorflow.keras.Model`.
    output_dir: (optional) `str`, the directory name to
      store the figures.
  """
  if not isinstance(history, tfk.callbacks.History):
    raise TypeError("`history` was expected to be of type "
                    "`tensorflow.keras.callbacks.History`, "
                    "but {} was provided.".format(type(history)))
  for metric in [k for k in history.history.keys() if not "val_" in k]:
    fig, ax = plt.subplots()
    ax.plot(history.history.get(metric), label="train")
    ax.plot(history.history.get("val_{}".format(metric)), label="validation")
    ax.set(title=metric, xlabel="epochs", **ax_set_kwargs)
    ax.legend()
    fig.tight_layout()
    if isinstance(output_dir, str):
      os.makedirs(output_dir, exist_ok=True)
      fig.savefig(
          os.path.join(output_dir, "{}.pdf".format(metric)),
          trasparent=True,
          dpi=300,
          format="pdf",
      )
    fig.show()


def leaderboard(
    benchmark: Text,
    results: Optional[Dict[Text, pd.DataFrame]] = None,
    output_dir: Optional[Text] = None,
    leaderboard_dir: Optional[Text] = None,
    **ax_set_kwargs,
):
  """Generates a leaderboard for all metrics in `benchmark`, by appending the
  (optional) `results`.

  Args:
    benchmark: `str`, the registerd name of `bdlb.Benchmark`.
    results: (optional) `dict`, dictionary of `pandas.DataFrames`
      with the results from a new method to be plotted against
      the leaderboard.
    leaderboard_dir: (optional) `str`, the path to the parent
      directory with all the leaderboard results.
    output_dir: (optional) `str`, the directory name to
      store the figures.
  """
  from .constants import BDLB_ROOT_DIR
  COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
  MARKERS = ["o", "D", "s", "8", "^", "*"]

  # The assumes path for stored baselines records
  leaderboard_dir = leaderboard_dir or os.path.join(BDLB_ROOT_DIR,
                                                    "leaderboard")
  benchmark_dir = os.path.join(leaderboard_dir, benchmark)
  if not os.path.exists(benchmark_dir):
    ValueError("No leaderboard data found at {}".format(benchmark_dir))

  # Metrics for which values are stored
  metrics = [
      x for x in os.listdir(benchmark_dir)
      if os.path.isdir(os.path.join(benchmark_dir, x))
  ]
  for metric in metrics:
    fig, ax = plt.subplots()
    # Iterate over baselines
    baselines = [
        x for x in os.listdir(os.path.join(benchmark_dir, metric))
        if ".csv" in x
    ]
    for b, baseline in enumerate(baselines):
      baseline = baseline.replace(".csv", "")
      # Fetch results
      df = pd.read_csv(
          os.path.join(benchmark_dir, metric, "{}.csv".format(baseline)))
      # Parse columns
      retained_data = df["retained_data"]
      mean = df["mean"]
      std = df["std"]
      # Visualize mean with standard error
      ax.plot(
          retained_data,
          mean,
          label=baseline,
          color=COLORS[b % len(COLORS)],
          marker=MARKERS[b % len(MARKERS)],
      )
      ax.fill_between(
          retained_data,
          mean - std,
          mean + std,
          color=COLORS[b % len(COLORS)],
          alpha=0.25,
      )
    if results is not None:
      # Plot results from dictionary
      if metric in results:
        df = results[metric]
        baseline = df.name if hasattr(df, "name") else "new_method"
        # Parse columns
        retained_data = df["retained_data"]
        mean = df["mean"]
        std = df["std"]
        # Visualize mean with standard error
        ax.plot(
            retained_data,
            mean,
            label=baseline,
            color=COLORS[(b + 1) % len(COLORS)],
            marker=MARKERS[(b + 1) % len(MARKERS)],
        )
        ax.fill_between(
            retained_data,
            mean - std,
            mean + std,
            color=COLORS[(b + 1) % len(COLORS)],
            alpha=0.25,
        )
    ax.set(xlabel="retained data", ylabel=metric)
    ax.legend()
    fig.tight_layout()
    if isinstance(output_dir, str):
      os.makedirs(output_dir, exist_ok=True)
      fig.savefig(
          os.path.join(output_dir, "{}.pdf".format(metric)),
          trasparent=True,
          dpi=300,
          format="pdf",
      )
