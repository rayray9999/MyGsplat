# gsplat

[![Core Tests.](https://github.com/nerfstudio-project/gsplat/actions/workflows/core_tests.yml/badge.svg?branch=main)](https://github.com/nerfstudio-project/gsplat/actions/workflows/core_tests.yml)
[![Docs](https://github.com/nerfstudio-project/gsplat/actions/workflows/doc.yml/badge.svg?branch=main)](https://github.com/nerfstudio-project/gsplat/actions/workflows/doc.yml)

[http://www.gsplat.studio/](http://www.gsplat.studio/)

gsplat is an open-source library for CUDA accelerated rasterization of gaussians with python bindings. It is inspired by the SIGGRAPH paper [3D Gaussian Splatting for Real-Time Rendering of Radiance Fields](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). This libary contains the neccessary components for efficient 3D to 2D projection, sorting, and alpha compositing of gaussians and their associated backward passes for inverse rendering.

This project was greatly inspired by original paper [3D Gaussian Splatting
for Real-Time Radiance Field Rendering
](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) by Kerbl* and Kopanas* et al. While building this library, we prioritized having a developer friendly Python API. As such, when this backend is plugged into the nerfstudio pipeline, it trains 5x~ slower than the original implementation. Please refer to the original [code release](https://github.com/graphdeco-inria/gaussian-splatting) for the optimized implementation. 

![Teaser](/docs/source/imgs/training.gif?raw=true)

## Installation

**Dependence**: Please install [Pytorch](https://pytorch.org/get-started/locally/) first.

The easist way is to install from PyPI. In this way it will build the CUDA code **on the first run** (JIT).

```bash
pip install gsplat
```

Or install from source. In this way it will build the CUDA code during installation.

```bash
pip install git+https://github.com/nerfstudio-project/gsplat.git
```

## Examples

Fit a 2D image with 3D Gaussians.
you can change change iterations or nums of points at main function in new_train.py

```bash
pip install -r examples/requirements.txt
python examples/new_train.py
```

## Use saved parameter

You can use the npz file saved in parameter.
And run examples/sample.py to do superresolution.
Remember to change the npz file in sample.py....




