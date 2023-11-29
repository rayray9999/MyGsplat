import cv2  
import numpy as np  
from skimage.metrics import structural_similarity as ssim  
  
# Load the two input images  
imageA = cv2.imread("examples/results/GS_1_0802x4_200000.png")  
imageB = cv2.imread("picture/0802x4.png")  
  
# Convert images to grayscale  
imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)  
imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)  
  
# Compute PSNR  
psnr = cv2.PSNR(imageA, imageB)  
  
# Compute SSIM  
ssim = ssim(imageA, imageB, data_range=imageB.max() - imageB.min())  
  
print("PSNR: ", psnr)  
print("SSIM: ", ssim)  
