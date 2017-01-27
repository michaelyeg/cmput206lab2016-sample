import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('damage_mask.bmp',0)
img1=cv2.imread('damaged_cameraman.bmp',0)

width =len(img)
height = len(img)
 
dst = cv2.GaussianBlur(img1,(5,5),0)
for i in range(0,10):
    dst2 = cv2.GaussianBlur(dst,(5,5),0)
    for j in range(0,height):
        for k in range(0,width):
            print img[j][k]
            if img[j][k]>=128:
                dst2[j][k]=img1[j][k]
    dst=dst2
    



plt.subplot(121),plt.imshow(img,cmap="gray"),plt.title('mask')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst,cmap="gray"),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()
