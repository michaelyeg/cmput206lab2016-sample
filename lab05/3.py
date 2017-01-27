import cv2
import numpy as np

def nothing(x):
    pass

# Create a black image, a window
img = cv2.imread('ex2.jpg',0)
cv2.namedWindow('image')
edge = cv2.Canny(img,100,100)

# create trackbars for color change
cv2.createTrackbar('threshold1','image',0,255,nothing)
cv2.createTrackbar('threshold2','image',0,255,nothing)


while(1):
    
    cv2.imshow('image',edge)
    
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    
    r = cv2.getTrackbarPos('threshold1','image')
    g = cv2.getTrackbarPos('threshold2','image')

    edge = cv2.Canny(img,r,g)
    
cv2.destroyAllWindows()
