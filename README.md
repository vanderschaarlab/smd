# SMD: Soft Mixture Denoising for Diffusion Models

The Gaussian denoising paradigm limits the approximation capability of current diffusion models. [The paper](https://openreview.net/forum?id=aaBnFAyW9O) in ICLR-2024 theoretically analyzes this problem and introduces a more expressive alternative: soft mixture denoising (SMD). This repository contains a Pytorch implementation of the diffusion model with SMD.

## Setup

Firstly, create a folder called "dataset", containing a set of fix-sized images. For example, 256 x 256 images from [CelebA-HQ](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Image formats of many kinds (e.g., jpg, png, and tiff) are supported.

Secondly, fork the repository and build a virtual environment with some necessary packages

```
$ conda create --name tmp_env python=3.8
$ conda activate tmp_env
$ pip install -r requirements.txt
```

## Run Scripts

Train a diffusion model with SMD with $1000$ denoising iterations and $128$ hidden units:

```
bash cases/run_smd.sh dataset 1000 128
```

Train a vanilla diffusion model with similar hyper-parameters:

```
bash cases/run_vanilla.sh dataset 1000 128
```

## Citation

```
@inproceedings{
li2024soft,
title={Soft Mixture Denoising: Beyond the Expressive Bottleneck of Diffusion Models},
author={Yangming Li and Boris van Breugel and Mihaela van der Schaar},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=aaBnFAyW9O}
}
```
