import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

# Read in the image
img = cv2.imread('test.jpg', 0)

# Obtain the dimensionality
h, w = img.shape[:2]

# Intensity level
K = 256

# Calculate the histogram
hist = cv2.calcHist([img],[0],None,[256],[0,256])

# Calculate the cumulative histogram
hist_cum = np.cumsum(hist)

# Flatten the image array so that only one loop is needed 
img1 = np.ravel(img)

# Initilize another image
img_eq = img1

# Update the pixel intensities in every location
for i in range(h * w):
    a = img1[i]
    img_eq[i] = math.floor((K - 1) / float(h * w) * hist_cum[a] + 0.5)

# Reshape the image back
img_eq = np.reshape(img_eq, (h, w))

# Calculate the histogram for the new image
hist_eq = cv2.calcHist([img_eq],[0],None,[256],[0,256])

plt.subplot(221), plt.imshow(cv2.imread('test.jpg'), 'gray'), plt.title('Original Image')
plt.subplot(222), plt.plot(hist), plt.title('Histogram')
plt.xlim([0,256])
plt.subplot(223), plt.imshow(img_eq, 'gray'), plt.title('New Image')
plt.subplot(224), plt.plot(hist_eq), plt.title('Histogram After Equalization')
plt.xlim([0,256])
plt.show()

