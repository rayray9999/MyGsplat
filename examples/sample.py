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
from simple_trainer import SimpleTrainer, image_path_to_tensor
from skimage.metrics import structural_similarity as ssim  
avg_psnr = 0
avg_ssim=0
up = 1
for i in range(801,802):
    #original_image = cv2.imread(f"../liif/load/DIV2K_valid_HR/{str(i).zfill(4)}.png")
    original_image = cv2.imread(f"picture/{str(i).zfill(4)}.png")
    #trainer = SimpleTrainer(gt_image=image_path_to_tensor(f"../liif/load/DIV2K_valid_LR_bicubic/X4/{str(i).zfill(4)}x4.png"), num_points=0,h_multi=1,w_multi=1)  
    trainer = SimpleTrainer(gt_image=image_path_to_tensor(f"../liif/load/DIV2K_valid_HR/{str(i).zfill(4)}.png"), num_points=0,h_multi=1,w_multi=1)  
    generated_image = trainer.rasterize_from_saved_parameters(f"parameter/blurred_penguin_100.npz")  
    #generated_image = trainer.rasterize_from_saved_parameters(f"parameter/{str(i).zfill(4)}x4.npz")  
    generated_image = (generated_image.detach().cpu().numpy() * 255).astype(np.uint8)
    generated_image = cv2.cvtColor(generated_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(os.getcwd(),f"examples/results/CtoF_{str(i).zfill(4)}.png"),generated_image)
    generated_image = cv2.cvtColor(generated_image, cv2.COLOR_RGB2GRAY)
    original_image =  cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    if up ==1:
    # Compute the PSNR  
        psnr = cv2.PSNR(original_image, generated_image)  
        SSIM = ssim(original_image, generated_image, data_range=generated_image.max() - original_image.min())
        avg_psnr+=psnr
        avg_ssim+=SSIM  
        print(psnr)
avg_ssim/=100
avg_psnr/=100
print("avg_PSNR",avg_psnr)
print("avg_SSIM",avg_ssim)