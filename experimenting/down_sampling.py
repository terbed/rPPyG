from skimage.transform import downscale_local_mean
import skimage.io as sio
import cv2
import numpy as np
import time
from matplotlib import pyplot as plt

patch_size = 25

rgb_img = sio.imread("orban.jpg")
cv2.imshow("Original", rgb_img)
cv2.waitKey(0)

start = time.time()
y = int(rgb_img.shape[0]/patch_size)
x = int(rgb_img.shape[1]/patch_size)
print(f"Number of subregions: {y*x}")

downsampled_image = np.empty(shape=(y, x, 3), dtype=np.float)
for i in range(y):
    for j in range(x):
        downsampled_image[i, j, :] = np.mean(rgb_img[i*patch_size:i*patch_size+patch_size-1, j*patch_size:j*patch_size+patch_size-1, :], axis=(0, 1))
end = time.time()
dur = (end-start)
print(dur)

# Downsample the image
start = time.time()
Id = downscale_local_mean(rgb_img, (patch_size, patch_size, 1))
end = time.time()
print(f"sklearn = {(end-start)}")

print(downsampled_image.shape)
print(Id.shape)
cv2.namedWindow("Downsampled", cv2.WINDOW_NORMAL)
cv2.imshow("Downsampled", downsampled_image.astype(int)*256)
cv2.resizeWindow("Downsampled", 500, 700)

# plt.figure(figsize=(10, 8))
# plt.imshow(downsampled_image.astype(int))
# plt.show()

cv2.namedWindow("Downsampled-sk", cv2.WINDOW_NORMAL)
cv2.imshow("Downsampled-sk", Id.astype(int)*256)
cv2.resizeWindow("Downsampled-sk", 500, 700)
cv2.waitKey(0)

"""
Result:
downscale a little faster and the result is the same
"""