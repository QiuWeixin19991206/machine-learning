import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img = cv.imread('E:\c++\pantyhose.jpg')
'''腐蚀和膨胀'''
# #创建核结构
# kernel = np.ones((10, 10), np.uint8)
# erosion = cv.erode(img, kernel)#腐蚀
# dilate = cv.dilate(img, kernel)#膨胀
# #图形有1行3列大小为宽度10英寸高度8英寸分辨率为100点每英寸
# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 8), dpi=300)
# axes[0].imshow(img)
# axes[0].set_title("原图")
# axes[1].imshow(erosion)
# axes[1].set_title("erosion")
# axes[2].imshow(dilate)
# axes[2].set_title("dilate")
# plt.show()
'''开闭运算'''
kernel = np.ones((20, 20), np.uint8)
cv2open = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
cv2close = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
limao = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
heimao = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 8), dpi=300)
axes[0].imshow(img)
axes[0].set_title("PHOTO")
axes[1].imshow(cv2open)
axes[1].set_title("open")
axes[2].imshow(cv2close)
axes[2].set_title("close")
axes[3].imshow(cv2open)
axes[3].set_title("TOPHAT")
axes[4].imshow(cv2close)
axes[4].set_title("BLACKHAT")
plt.show()





