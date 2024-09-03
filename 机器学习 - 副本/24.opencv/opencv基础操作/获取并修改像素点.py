import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img = cv.imread("1.png", 1)
#通道拆分
b, g, r = cv.split(img)
#通道合并
img = cv.merge((b, g, r))
#显示结果
# plt.imshow(b, cmap=plt.cm.gray)
# plt.show()
# plt.imshow(img[:, :, ::-1])
# plt.show()
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
plt.imshow(gray, cmap=plt.cm.gray)
plt.show()

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
plt.imshow(hsv)
plt.show()