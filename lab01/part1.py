import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read in the image
img = cv2.imread('test.jpg', 0)

# Obtain the dimensionality
h, w = img.shape[:2]

# Using opencv function to calculate the histogram
hist_cv = cv2.calcHist([img],[0],None,[256],[0,256])

# Using our own function to calculate the histogram

# Initialize the histogram
hist_my = np.zeros(256)

# Flatten the image array so that only one loop is needed 
img1 = np.ravel(img)

# Go over all the pixels in the image to increase the frequency properly
for i in range(h * w):
    hist_my[img1[i]] += 1

# Show the two histograms to compare
plt.subplot(121), plt.plot(hist_cv), plt.title('Histogram by OpenCV')
plt.xlim([0,256])
plt.subplot(122), plt.plot(hist_my), plt.title('Our Own Histogram')
plt.xlim([0,256])
plt.show()
