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
up =1
frames= []
H = 300
W = 500
for T in range(10,41):
    times = T/10.0
    height = round(H *times)
    width  =round(W*times)
    print(height,width)
    #original_image = cv2.imread(f"picture/{pic_name}.png")
    gt_image = torch.ones((height, width , 3)) * 1.0
        # make top left and bottom right red, blue
    gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0])
    gt_image[height // 2 :, width // 2 :, :] = torch.tensor([0.0, 0.0, 1.0])
    trainer = SimpleTrainer(gt_image=gt_image, num_points=0,h_multi=1,w_multi=1)  
    generated_image = trainer.rasterize_from_saved_parameters(f"parameter/0801x4.npz")  
    generated_image = (generated_image.detach().cpu().numpy() * 255).astype(np.uint8)
    frames.append(generated_image)
    print(height)
frames = [Image.fromarray(frame) for frame in frames]
out_dir = os.path.join(os.getcwd(), "coool")
os.makedirs(out_dir, exist_ok=True)
frames[0].save(
                f"{out_dir}/test.gif",
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=5,
                loop=0,
            )