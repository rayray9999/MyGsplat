import math
import os
import time
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import tyro
from gsplat.project_gaussians import ProjectGaussians
from gsplat.rasterize import RasterizeGaussians
from PIL import Image
from torch import Tensor, optim
import cv2
from torch.optim.lr_scheduler import StepLR  
from utils.loss_utils import l1_loss, ssim
from scene.Gaussian_model import GaussianModel
pic_name = "blurred_penguin_100"
Blur_num = 100


class SimpleTrainer:
    """Trains random gaussians to fit an image."""
    def save_parameters(self, filename):      
        try:    
            parameters = {      
                'means': self.gaussian_model.means.detach().cpu().numpy(),      
                'scales': self.gaussian_model.scales.detach().cpu().numpy(),      
                'quats': self.gaussian_model.quats.detach().cpu().numpy(),      
                'rgbs': self.gaussian_model.rgbs.detach().cpu().numpy(),      
                'opacities': self.gaussian_model.opacities.detach().cpu().numpy(),      
            }      
            np.savez(filename, **parameters)      
        except Exception as e:    
            print(f"Failed to save parameters: {e}")    
  
    def rasterize_from_saved_parameters(self, filename):    
        loaded_parameters = np.load(filename)    
        self.gaussian_model.means = torch.tensor(loaded_parameters['means'], device=self.device)    
        self.gaussian_model.scales = torch.tensor(loaded_parameters['scales'], device=self.device)    
        self.gaussian_model.quats = torch.tensor(loaded_parameters['quats'], device=self.device)    
        self.gaussian_model.rgbs = torch.tensor(loaded_parameters['rgbs'], device=self.device)    
        self.gaussian_model.opacities = torch.tensor(loaded_parameters['opacities'], device=self.device)     
    
        return self.forward_new()  # or self.forward_slow()    
    
    def forward_new(self):  
        xys, depths, radii, conics, num_tiles_hit, cov3d = ProjectGaussians.apply(  
            self.gaussian_model.means,  
            self.gaussian_model.scales,  
            1,  
            self.gaussian_model.quats / self.gaussian_model.quats.norm(dim=1, keepdim=True),  
            self.viewmat,  
            self.viewmat,  
            self.focal,  
            self.focal,  
            self.W / 2,  
            self.H / 2,  
            self.H,  
            self.W,  
            self.tile_bounds,  
        )  
    
        return RasterizeGaussians.apply(  
            xys,  
            depths,  
            radii,  
            conics,  
            num_tiles_hit,  
            torch.sigmoid(self.gaussian_model.rgbs),  
            torch.sigmoid(self.gaussian_model.opacities),  
            self.H,  
            self.W,  
        )  

    def __init__(
        self,
        gt_image: Tensor,
        num_points: int = 5000,
        h_multi = 1,
        w_multi =1
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = gt_image.to(device=self.device)
        self.gaussian_model = GaussianModel(num_points, self.device)

        BLOCK_X, BLOCK_Y = 16, 16
        fov_x = math.pi / 2.0
        self.H, self.W = gt_image.shape[0]*h_multi, gt_image.shape[1]*w_multi
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        self.tile_bounds = (
            (self.W + BLOCK_X - 1) // BLOCK_X,
            (self.H + BLOCK_Y - 1) // BLOCK_Y,
            1,
        )
        self.img_size = torch.tensor([self.W, self.H, 1], device=self.device)
        self.block = torch.tensor([BLOCK_X, BLOCK_Y, 1], device=self.device)
        self._init_viewmat()

        # add from 3D GS
        self.lambda_dssim = 0.2


    def _init_viewmat(self):
        """Random gaussians"""

        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )
        self.viewmat.requires_grad = False

    def train(self, iterations: int = 2000, lr: float = 0.001, save_imgs: bool = False):
        self.gaussian_model.training_setup()
        #mse_loss = torch.nn.MSELoss()
        frames = []
        times = [0] * 3  # project, rasterize, backward
        for iter in range(iterations):
            start = time.time()
            xys, depths, radii, conics, num_tiles_hit, cov3d = ProjectGaussians.apply(  
            self.gaussian_model.means,  
            self.gaussian_model.scales,  
            1,  
            self.gaussian_model.quats,  
            self.viewmat,  
            self.viewmat,  
            self.focal,  
            self.focal,  
            self.W / 2,  
            self.H / 2,  
            self.H,  
            self.W,  
            self.tile_bounds,  
        )
            torch.cuda.synchronize()
            times[0] += time.time() - start
            start = time.time()
            out_img = RasterizeGaussians.apply(  
            xys,  
            depths,  
            radii,  
            conics,  
            num_tiles_hit,  
            torch.sigmoid(self.gaussian_model.rgbs),  
            torch.sigmoid(self.gaussian_model.opacities),  
            self.H,  
            self.W,  
        )
            self.gaussian_model.update_learning_rate(iter)
            torch.cuda.synchronize()
            times[1] += time.time() - start
            #loss = mse_loss(out_img, self.gt_image)
            Ll1 = l1_loss(out_img, self.gt_image)
            loss = (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * (1.0 - ssim(out_img, self.gt_image))
            self.gaussian_model.optimizer.zero_grad()
            start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            times[2] += time.time() - start
            self.gaussian_model.optimizer.step() 
            print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")

            if save_imgs and iter % 100 == 0:
                global Blur_num
                if iterations-iter ==400: Blur_num=3
                if iterations-iter ==200: Blur_num=0
                self.gt_image = image_path_to_tensor(f'./picture/blurred_penguin_{Blur_num}.png').to(device=self.device)
                #frames.append((out_img.detach().cpu().numpy() * 255).astype(np.uint8))
                if Blur_num >5: Blur_num -= 5
                
        cv2.imwrite("find_large_Gaussian.png",cv2.cvtColor((out_img.detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        self.save_parameters(f"parameter/{pic_name}.npz")  
        if save_imgs == False:
            # save them as a gif with PIL
            frames = [Image.fromarray(frame) for frame in frames]
            out_dir = os.path.join(os.getcwd(), "renders")
            os.makedirs(out_dir, exist_ok=True)
            frames[0].save(
                f"{out_dir}/{pic_name}.gif",
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=5,
                loop=0,
            )
        print(
            f"Total(s):\nProject: {times[0]:.3f}, Rasterize: {times[1]:.3f}, Backward: {times[2]:.3f}"
        )
        print(
            f"Per step(s):\nProject: {times[0]/iterations:.5f}, Rasterize: {times[1]/iterations:.5f}, Backward: {times[2]/iterations:.5f}"
        )


def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor


def main(
    height: int = 256,
    width: int = 256,
    num_points: int = 800,
    save_imgs: bool = True,
    img_path: Optional[Path] = f'./picture/{pic_name}.png',
    iterations: int = 4200,
    lr: float = 0.01,
) -> None:
    global pic_name
    if img_path:
        # Get the last part of the path  
        base_name = os.path.basename(img_path)  
        
        # Remove the '.png' extension  
        pic_name = str(base_name.rsplit('.', 1)[0])
        gt_image = image_path_to_tensor(img_path)
    else:
        gt_image = torch.ones((height, width, 3)) * 1.0
        # make top left and bottom right red, blue
        gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0])
        gt_image[height // 2 :, width // 2 :, :] = torch.tensor([0.0, 0.0, 1.0])

    trainer = SimpleTrainer(gt_image=gt_image, num_points=num_points)
    trainer.train(
        iterations=iterations,
        lr=lr,
        save_imgs=save_imgs,
    )


if __name__ == "__main__":
    try:  
        tyro.cli(main)  
    except Exception as e:  
        print(f"An exception occurred: {e}")  

