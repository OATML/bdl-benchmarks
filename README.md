# Bayesian Deep Learning Benchmarks

**This repository is no longer being updated.** 

Please refer to the [Diabetic Retinopathy Detection implementation in Google's 'uncertainty-baselines' repo](https://github.com/google/uncertainty-baselines/tree/master/baselines/diabetic_retinopathy_detection) for up-to-date baseline implementations.

## Overview
In order to make real-world difference with **Bayesian Deep Learning** (BDL) tools, the tools must scale to real-world settings. And for that we, the research community, must be able to evaluate our inference tools (and iterate quickly) with real-world benchmark tasks. We should be able to do this without necessarily worrying about application-specific domain knowledge, like the expertise often required in medical applications for example. We require benchmarks to test for inference robustness, performance, and accuracy, in addition to cost and effort of development. These benchmarks should be at a variety of scales, ranging from toy `MNIST`-scale benchmarks for fast development cycles, to large data benchmarks which are truthful to real-world applications, capturing their constraints.

Our BDL benchmarks should 
* provide a transparent, modular and consistent interface for the evaluation of deep probabilistic models on a variety of _downstream tasks_;
* rely on expert-driven metrics of uncertainty quality (actual applications making use of BDL uncertainty in the real-world), but abstract away the expert-knowledge and eliminate the boilerplate steps necessary for running experiments on real-world datasets;
* make it easy to compare the performance of new models against _well tuned baselines_, models that have been well-adopted by the machine learning community, under a fair and realistic setting (e.g., computational resources, model sizes, datasets);
* provide reference implementations of baseline models (e.g., Monte Carlo Dropout Inference, Mean Field Variational Inference, Deep Ensembles), enabling rapid prototyping and easy development of new tools;
* be independent of specific deep learning frameworks (e.g., not depend on `TensorFlow`, `PyTorch`, etc.), and integrate with the SciPy ecosystem (i.e., `NumPy`, `Pandas`, `Matplotlib`). Benchmarks are framework-agnostic, while baselines are framework-dependent.

In this repo we strive to provide such well-needed benchmarks for the BDL community, and collect and maintain new baselines and benchmarks contributed by the community. **A colab notebook demonstrating the MNIST-like workflow of our benchmarks is [available here](notebooks/diabetic_retinopathy_diagnosis.ipynb)**.

**We highly encourage you to contribute your models as new *baselines* for others to compete against, as well as contribute new *benchmarks* for others to evaluate their models on!**

## List of Benchmarks

**Bayesian Deep Learning Benchmarks** (BDL Benchmarks or `bdlb` for short), is an open-source framework that aims to bridge the gap between the design of deep probabilistic machine learning models and their application to real-world problems. Our currently supported benchmarks are:

- [x] [Diabetic Retinopathy Diagnosis](baselines/diabetic_retinopathy_diagnosis) (in [`alpha`](https://github.com/OATML/bdl-benchmarks/tree/alpha/), following [Leibig et al.](https://www.nature.com/articles/s41598-017-17876-z))
  - [x] [Deterministic](baselines/diabetic_retinopathy_diagnosis/deterministic)
  - [x] [Monte Carlo Dropout](baselines/diabetic_retinopathy_diagnosis/mc_dropout) (following [Gal and Ghahramani, 2015](https://arxiv.org/abs/1506.02142))
  - [x] [Mean-Field Variational Inference](baselines/diabetic_retinopathy_diagnosis/mfvi) (following [Peterson and Anderson, 1987](https://pdfs.semanticscholar.org/37fa/18c66b8130b9f9748d9c94472c5671fb5622.pdf), [Wen et al., 2018](https://arxiv.org/abs/1803.04386))
  - [x] [Deep Ensembles](baselines/diabetic_retinopathy_diagnosis/deep_ensembles) (following [Lakshminarayanan et al., 2016](https://arxiv.org/abs/1612.01474))
  - [x] [Ensemble MC Dropout](baselines/diabetic_retinopathy_diagnosis/deep_ensembles) (following [Smith and Gal, 2018](https://arxiv.org/abs/1803.08533))

- [ ] Autonomous Vehicle's Scene Segmentation (in `pre-alpha`, following [Mukhoti et al.](https://arxiv.org/abs/1811.12709))
- [ ] Galaxy Zoo (in `pre-alpha`, following [Walmsley et al.](https://arxiv.org/abs/1905.07424))
- [ ] Fishyscapes (in `pre-alpha`, following [Blum et al.](https://arxiv.org/abs/1904.03215))


## Installation

*BDL Benchmarks* is shipped as a PyPI package (Python3 compatible) installable as:

```
pip3 install git+https://github.com/OATML/bdl-benchmarks.git
```

The data downloading and preparation is benchmark-specific, and you can follow the relevant guides at `baselines/<benchmark>/README.md` (e.g. [`baselines/diabetic_retinopathy_diagnosis/README.md`](baselines/diabetic_retinopathy_diagnosis/README.md)).


## Examples

For example, the [Diabetic Retinopathy Diagnosis](baselines/diabetic_retinopathy_diagnosis) benchmark comes with several baselines, including MC Dropout, MFVI, Deep Ensembles, and more. These models are trained with images of blood vessels in the eye:

<p align="center">
<img src="http://www.cs.ox.ac.uk/people/angelos.filos/assets/bdl-benchmarks/samples.png" >
</p>

The models try to predict diabetic retinopathy, and use their uncertainty for *prescreening* (sending patients the model is uncertain about to an expert for further examination). When you implement a new model, you can easily benchmark your model against existing baseline results provided in the repo, and generate plots using expert metrics (such as the AUC of retained data when referring 50% most uncertain patients to an expert):

<p align="center">
<img src="http://www.cs.ox.ac.uk/people/angelos.filos/assets/bdl-benchmarks/metrics/auc.png" style="float: left; width: 25%; margin-right: 1%; margin-bottom: 0.5em; margin-top: 0.0em" >
<img src="http://www.cs.ox.ac.uk/people/angelos.filos/assets/bdl-benchmarks/metrics/accuracy.png" style="float: left; width: 25%; margin-right: 1%; margin-bottom: 0.5em; margin-top: 0.0em" >
<img src="http://www.cs.ox.ac.uk/people/angelos.filos/assets/bdl-benchmarks/metrics/legend.png" style="float: left; width: 100%; margin-bottom: 0.5em; margin-top: 0.0em" >
</p>

**You can even play with a [colab notebook](notebooks/diabetic_retinopathy_diagnosis.ipynb) to see the workflow of the benchmark**, and contribute your model for others to benchmark against. 


## Cite as

Please cite individual benchmarks when you use these, as well as the baselines you compare against. For the [Diabetic Retinopathy Diagnosis](baselines/diabetic_retinopathy_diagnosis) benchmark please see [here](baselines/diabetic_retinopathy_diagnosis#cite-as).

## Acknowledgements

The repository is developed and maintained by the [Oxford Applied and Theoretical Machine Learning](http://oatml.cs.ox.ac.uk/) group, with sponsorship from:

<table align="center">
  <tr>
    <td><img src="https://github.com/OATML/bdl-benchmarks/blob/alpha/assets/intel.png" style="float: left; width: 200px; margin-right: 1%; margin-bottom: 0.5em; margin-top: 0.0em"></td>
    <td><img src="https://github.com/OATML/bdl-benchmarks/blob/alpha/assets/oatml.png" style="float: left; width: 200px; margin-right: 1%; margin-bottom: 0.5em; margin-top: 0.0em"></td>
    <td><img src="https://github.com/OATML/bdl-benchmarks/blob/alpha/assets/oxcs.png" style="float: left; width: 200px; margin-right: 1%; margin-bottom: 0.5em; margin-top: 0.0em"></td>
    <td><img src="https://github.com/OATML/bdl-benchmarks/blob/alpha/assets/turing.png" style="float: left; width: 200px; margin-right: 1%; margin-bottom: 0.5em; margin-top: 0.0em"></td>
  </tr>
 </table>
 
## Contact Us

Email us for questions at oatml@cs.ox.ac.uk, or submit any issues to improve the framework.
