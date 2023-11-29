import math
import os
import time
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import torch
import tyro
from gsplat.project_gaussians import ProjectGaussians
from gsplat.rasterize import RasterizeGaussians
from PIL import Image
from torch import Tensor, optim
from simple_trainer import main
print(os.getcwd())
for i in range (801,901):
    image_path = f"../liif/load/DIV2K_valid_LR_bicubic/X4/{str(i).zfill(4)}x4.png"
    MAIN=main(
    height= 256,
    width= 256,
    num_points= 100000,
    save_imgs = False,
    img_path = image_path,
    iterations = 4000,
    lr= 0.01,
)
    