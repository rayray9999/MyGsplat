import cv2  
import numpy as np  
  
# Load the image  
image = cv2.imread('./picture/0801x4.png')  
  
# Apply Gaussian blur  
blurred_image = cv2.GaussianBlur(image, (3,3), 0)  
    
    # Save the blurred image  
cv2.imwrite(f'picture/blurred_penguin_{3}.png', blurred_image)   
