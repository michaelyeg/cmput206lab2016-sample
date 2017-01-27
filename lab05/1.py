import cv2
import numpy as np
import math

ix,iy = -1,-1

# mouse callback function
def lblur(event,x,y,flags,param):
    global ix,iy,img,mask

    if event == cv2.EVENT_LBUTTONDOWN:
        x0,y0 = len(img),len(img[0])
        ix,iy = x0-y,y0-x

    elif event == cv2.EVENT_LBUTTONUP:
        blur = GBlur(img)
        for i in range(len(img)):
            for j in range(len(img[0])):
                img[i][j]=mask[i+ix][j+iy]*blur[i][j]+(1-mask[i+ix][j+iy])*img[i][j]
                
def GBlur(img):
    guss = cv2.GaussianBlur(img,(37,37),0)
    return guss

def gen_mask(img):
    x0,y0 = len(img),len(img[0])
    
    st = 40

    w = [[0 for _ in range(2*len(img[0]))] for _ in range(2*len(img))]

    for x in range(len(w)):
        for y in range(len(w[0])):
            try:
                w[x][y]=math.exp(-((x-x0)**2+(y-y0)**2)/st**2)
            except:
                continue
    return w

img = cv2.imread('ex1.jpg',0)
cv2.namedWindow('image')
cv2.setMouseCallback('image',lblur)
mask = gen_mask(img)

while(1):
    cv2.imshow('image',img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
    
cv2.destroyAllWindows()

    
