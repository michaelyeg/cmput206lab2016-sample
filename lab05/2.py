import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

img = cv2.imread('ex2.jpg',0)

kernel_x = np.array(
    [[-1,0,1],
     [-2,0,2],
     [-1,0,1]])
kernel_y = np.array(
    [[-1,-2,-1],
     [0,0,0],
     [1,2,1]])

kernel_img_x=cv2.filter2D(img,-1,kernel_x)
kernel_img_y=cv2.filter2D(img,-1,kernel_y)

sobel_x = cv2.Sobel(img,-1,1,0,ksize=3)
sobel_y = cv2.Sobel(img,-1,0,1,ksize=3)

sobel_grad = cv2.add(sobel_x,sobel_y)

#cv2.imshow('x',kernel_img_x)

plt.imshow(kernel_img_x,cmap="gray")
plt.show()
plt.imshow(kernel_img_y,cmap="gray")
plt.show()
plt.imshow(sobel_grad,cmap="gray")
plt.show()
