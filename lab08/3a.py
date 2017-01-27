import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('redbloodcell.jpg', 0)

#find the black
ret1,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#delete black on img
for i in range(len(img)):
    for j in range(len(img[0])):
        if thresh1[i][j]<ret1:
            img[i][j]=255

#gray img
ret2,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#change black on thresh2 to gary
for i in range(len(img)):
    for j in range(len(img[0])):
        if thresh2[i][j]<ret1:
            thresh2[i][j]= 127
            
#combin thresh1 and thresh2         
for i in range(len(img)):
    for j in range(len(img[0])):
        if thresh2[i][j]>ret2:
            thresh2[i][j]= thresh1[i][j]


cv2.imshow("image",thresh2)
cv2.waitKey(0)
cv2.destroyAllWindows()