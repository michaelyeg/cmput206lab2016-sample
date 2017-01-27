import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('100.jpg')

kernel = np.matrix('-1 -1 -1, -1 8 -1 , -1 -1 -1')
dst = cv2.filter2D(img,-1,kernel)
a=img+dst
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(a),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()
