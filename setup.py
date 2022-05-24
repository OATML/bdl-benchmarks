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

import os

from setuptools import find_packages
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
  long_description = f.read()

setup(
    name="bdlb",
    version="0.0.2",
    description="BDL Benchmarks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/oatml/bdl-benchmarks",
    author="Oxford Applied and Theoretical Machine Learning Group",
    author_email="oatml@googlegroups.com",
    license="Apache-2.0",
    packages=find_packages(),
    install_requires=[
        "numpy==1.18.5",
        "scipy==1.4.1",
        "pandas==1.0.4",
        "matplotlib==3.2.1",
        "seaborn==0.10.1",
        "scikit-learn==0.21.3",
        "kaggle==1.5.6",
        "opencv-python==4.2.0.34",
        "tensorflow-gpu==2.6.4",
        "tensorflow-probability==0.7.0",
        "tensorflow-datasets==1.1.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Researchers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: Apache 2.0 License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
