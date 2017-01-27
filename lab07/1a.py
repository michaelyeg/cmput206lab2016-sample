import cv2
import numpy as np

img = cv2.imread('PeppersBayerGray.bmp',0)
img2 = np.array(img, dtype = 'float32')

w,h = len(img2[0]),len(img2)

def green(a,b):
    G = np.zeros((a,b),np.float32)
    for i in range(a-1):  
        for j in range(b-1):
            if i == 0 and j%2 == 1:
                G[i][j] = (img2[i][j-1]+img2[i][j+1])/2 #b
            elif i == 0 and j % 3 == 0:
                G[i][j] = (img2[i][j-1]+img2[i+1][j])/2 #d
            elif i % 3 == 0 and j == 0:
                G[i][j] = (img2[i-1][j]+img2[i][j+1])/2 #m
            elif i % 2 == 1 and j == 0:
                G[i][j] = (img2[i-1][j]+img2[i+1][j])/2 #e     
            elif (i % 2 == 0 and j % 2 == 1) or (i % 2 == 1 and j % 2 == 0):
                G[i][j] = (img2[i-1][j]+img2[i+1][j]+img2[i][j-1]+img2[i][j+1])/4 #g,j
    return G

def red(a,b):
    R = np.zeros((a,b),np.float32)
    for i in range(a-1):
        for j in range(b):
            if i == 0 and j == 0:
                R[i][j] = img2[i][j+1] #a
            elif i % 2 == 0 and j % 2 == 0:
                R[i][j] = (img2[i][j-1]+img2[i][j+1])/2 #c
            elif i % 2 == 1 and j == 0:
                R[i][j] = (img2[i-1][j+1]+img2[i+1][j+1])/2 #e
            elif i % 2 ==1 and j % 2 == 1:
                R[i][j] = (img2[i-1][j]+img2[i+1][j])/2 #f
            elif i % 2 == 1 and j % 2 == 0:
                R[i][j] = (img2[i-1][j-1]+img2[i-1][j+1]+img2[i+1][j-1]+img2[i+1][j+1])/4 #g
    return R

def blue(a,b):
    B = np.zeros((a,b),np.float32)
    for i in range(a-1):
        for j in range(b-1):
            if i == 0 and j % 2 ==0:
                B[i][j]= img2[i+1][j] 
            elif i % 2 == 0 and j % 2 == 1:
                B[i][j] = ((img2[i-1][j-1]+img2[i-1][j+1]+img2[i+1][j-1]+img2[i+1][j+1])/4) #j
            elif i % 2 == 0 and j % 2 == 0:
                B[i][j] = ((img2[i-1][j]+img2[i+1][j])/2) #i
            elif i % 2 == 1 and j % 2 == 1:
                B[i][j] = ((img2[i][j-1]+img2[i][j+1])/2) #f
    return B
   
out = np.zeros((h,w,3),dtype=np.uint8)

out[:,:,0]= blue(h,w)
out[:,:,1] = green(h,w)
out[:,:,2] = red(h,w)

cv2.imshow('Basic Bayer', out)
cv2.waitKey()
cv2.destroyAllWindows()

