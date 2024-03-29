import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read in the image
img1 = cv2.imread('day.jpg', 0)

# Read in another image
img2 = cv2.imread('night.jpg', 0)

# Calculate the histograms
hist1 = cv2.calcHist([img1],[0],None,[256],[0,256])
hist2 = cv2.calcHist([img2],[0],None,[256],[0,256])

# Normalize the histograms
hist1_norm = hist1 / sum(hist1)
hist2_norm = hist2 / sum(hist2)

# Calculate the Bhattacharya coefficient
# When programming, try to avoid using unnecessary loops
bc = sum(np.sqrt(np.multiply(hist1_norm, hist2_norm)))

print bc
