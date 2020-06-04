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
"""Default values for some parameters of the API when no values are passed."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Root directory of the BDL Benchmarks module.
BDLB_ROOT_DIR: str = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))

# Directory where to store processed datasets.
DATA_DIR: str = os.path.join(BDLB_ROOT_DIR, "data")

# URL to hosted datasets
DIABETIC_RETINOPATHY_DIAGNOSIS_URL_MEDIUM = "https://drive.google.com/uc?id=1WAvS-pQsVLxUJiClmKLnVNQkoKmRt2I_"

# URL to hosted assets
LEADERBOARD_DIR_URL: str = "https://drive.google.com/uc?id=1LQeAfqMQa4lot09qAuzWa3t6attmCeG-"
