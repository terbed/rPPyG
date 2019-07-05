import cv2
import numpy as np

image = cv2.imread("orban.jpg")
cv2.imshow("Loaded image", image)
cv2.waitKey(0)

blurred_image = cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=5)
cv2.imshow("Blurred image", blurred_image)
cv2.waitKey(0)

cv2.destroyAllWindows()