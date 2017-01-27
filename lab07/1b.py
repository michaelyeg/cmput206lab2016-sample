import cv2
import numpy as np

img = cv2.imread('PeppersBayerGray.bmp',0)
img2 = np.array(img, dtype = 'int64')

w,h = len(img2[0]), len(img2)

def green(a,b):
    G = np.zeros((a,b),np.float32)
    for i in range(a-1):  
        for j in range(b-1):
            if i == 0 and j%2 == 1:
                G[i][j] = (img2[i][j-1]+img2[i][j+1])/2
            elif i == 0 and j % 3 == 0:
                G[i][j] = (img2[i][j-1]+img2[i+1][j])/2
            elif i % 3 == 0 and j == 0:
                G[i][j] = (img2[i-1][j]+img2[i][j+1])/2
            elif i % 2 == 1 and j == 0:
                G[i][j] = (img2[i-1][j]+img2[i+1][j])/2         
            elif (i % 2 == 0 and j % 2 == 1) or (i % 2 == 1 and j % 2 == 0):
                G[i][j] = (img2[i-1][j]+img2[i+1][j]+img2[i][j-1]+img2[i][j+1])/4
    return G


def red(a,b):
    R = np.zeros((a,b),np.float32)
    for i in range(a-1):
        for j in range(b):
            if j == 0 and i == 0:
                R[i][j] = img2[i][j+1]
            elif j % 2 == 0 and i % 2 == 0:
                R[i][j] = (img2[i][j-1]+img2[i][j+1])/2
            elif i % 2 == 1 and j == 0:
                R[i][j] = (img2[i-1][j+1]+img2[i+1][j+1])/2
            elif i % 2 ==1 and j % 2 == 1:
                R[i][j] = (img2[i-1][j]+img2[i+1][j])/2
            elif i % 2 == 1 and j % 2 == 0:
                R[i][j] = (img2[i-1][j-1]+img2[i-1][j+1]+img2[i+1][j-1]+img2[i+1][j+1])/4
    return R

def blue(a,b):
    B = np.zeros((a,b),np.float32)
    for i in range(a-1):
        for j in range(b-1):
            if i == 0 and j % 2 ==0:
                B[i][j]= img2[i+1][j]
            elif i % 2 == 0 and j % 2 == 1:
                B[i][j] = ((img2[i-1][j-1]+img2[i-1][j+1]+img2[i+1][j-1]+img2[i+1][j+1])/4)
            elif i % 2 == 0 and j % 2 == 0:
                B[i][j] = ((img2[i-1][j]+img2[i+1][j])/2)
            elif i % 2 == 1 and j % 2 == 1:
                B[i][j] = ((img2[i][j-1]+img2[i][j+1])/2)
    return B

IR=red(h,w)
IG=green(h,w)
IB=blue(h,w)

DR = IR-IG
DB = IB-IG
MR = cv2.medianBlur(DR,3)
MB = cv2.medianBlur(DB,3)
IRR = MR+IG
IBB = MB+IG
minIRR = np.amin(IRR)
minIBB = np.amin(IBB)

IRR = IRR-minIRR
IBB = IBB-minIBB
maxIRR = np.amax(IRR)
maxIBB = np.amax(IBB)

IRR = IRR*(255/maxIRR)
IBB = IBB*(255/maxIBB)

                      
out = np.zeros((h,w,3),dtype=np.uint8)
out[:,:,0] = IBB
out[:,:,1] = IG
out[:,:,2]= IRR
cv2.imshow('Advanced Bayer', out)
cv2.waitKey()
cv2.destroyAllWindows()
