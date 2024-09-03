import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img = cv.imread('E:\c++\pantyhose.jpg', 0)
'''边缘检测'''
# x = cv.Sobel(img, cv.CV_16S, 1, 0, ksize=-1)#计算卷积结果
# y = cv.Sobel(img, cv.CV_16S, 0, 1, ksize=-1)
# Scale_absX = cv.convertScaleAbs(x)#转换
# Scale_absY = cv.convertScaleAbs(y)
# img2 = cv.addWeighted(Scale_absY, 0.5, Scale_absY, 0.5, 0)#结果合成
# plt.figure(figsize=(10, 8), dpi=100)
# plt.subplot(121), plt.imshow(img, cmap=plt.cm.gray), plt.title('photo')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(img, cmap=plt.cm.gray), plt.title('bianyuan jiance')
# plt.xticks([]), plt.yticks([])
# plt.show()
'''laplacian转换'''
# result = cv.Laplacian(img, cv.CV_16S)
# Scale_abs = cv.convertScaleAbs(result)
# # 3 图像展示
# plt.figure(figsize=(10,8),dpi=100)
# plt.subplot(121),plt.imshow(img,cmap=plt.cm.gray),plt.title('原图')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(Scale_abs,cmap = plt.cm.gray),plt.title('Laplacian检测后结果')
# plt.xticks([]), plt.yticks([])
# plt.show()
'''Canny边缘检测'''
lowThreshold = 0
max_lowThreshold = 100
canny = cv.Canny(img, lowThreshold, max_lowThreshold)
# 3 图像展示
plt.figure(figsize=(10,8),dpi=100)
plt.subplot(121),plt.imshow(img,cmap=plt.cm.gray),plt.title('原图')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(canny,cmap = plt.cm.gray),plt.title('Canny检测后结果')
plt.xticks([]), plt.yticks([])
plt.show()






