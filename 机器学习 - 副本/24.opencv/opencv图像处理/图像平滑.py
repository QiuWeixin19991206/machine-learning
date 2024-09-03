import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# img = cv.imread('E:\c++\pantyhose.jpg')

'''均值滤波'''
# img1 = cv.blur(img, (5, 5))
'''高斯滤波'''
# img2 = cv.GaussianBlur(img, (3, 3), 1)
'''中值滤波'''
# img3 = cv.medianBlur(img, 5)
#绘图
# fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 8), dpi=300)
# axes[0].imshow(img)
# axes[0].set_title("img")
# axes[1].imshow(img1)
# axes[1].set_title("img1")
# axes[2].imshow(img2)
# axes[2].set_title("img2")
# axes[2].imshow(img3)
# axes[2].set_title("img3")
# plt.show()
# plt.figure(figsize=(10, 8), dpi=100)
# plt.subplot(121), plt.imshow(img[:, :, ::-1]), plt.title('原图')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(img3[:, :, ::-1]), plt.title('中值滤波')
# plt.xticks([]), plt.yticks([])
# plt.show()
'''直方图'''
img = cv.imread('E:\c++\pantyhose.jpg', 0)
# plt.imshow(img, cmap=plt.cm.gray)
# plt.show()
# img4 = cv.calcHist([img], [0], None, [256], [0, 256])
# plt.figure(figsize=(10, 8))
# plt.plot(img4)
# plt.show()
'''掩模'''
# 2. 创建蒙版
# mask = np.zeros(img.shape[:2], np.uint8)
# mask[400:650, 200:500] = 1#>=1就行
# # 3.掩模
# masked_img = cv.bitwise_and(img, img, mask=mask)
# # 4. 统计掩膜后图像的灰度图
# mask_histr = cv.calcHist([img], [0], mask, [256], [1, 256])
# # 5. 图像展示
# fig,axes=plt.subplots(nrows=2,ncols=2, figsize=(10,8))
# axes[0, 0].imshow(img, cmap=plt.cm.gray)#plt.cm.gray以灰度图显示
# axes[0, 0].set_title("原图")
# axes[0, 1].imshow(mask, cmap=plt.cm.gray)
# axes[0, 1].set_title("蒙版数据")
# axes[1, 0].imshow(masked_img, cmap=plt.cm.gray)
# axes[1, 0].set_title("掩膜后数据")
# axes[1, 1].plot(mask_histr)
# axes[1, 1].grid()
# axes[1, 1].set_title("灰度直方图")
# plt.show()
'''直方图均衡化'''
# dst = cv.equalizeHist(img)
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8), dpi=300)
# axes[0].imshow(img, cmap=plt.cm.gray)
# axes[0].set_title("img")
# axes[1].imshow(dst, cmap=plt.cm.gray)
# axes[1].set_title("dst")
# plt.show()
'''自适应直方图均衡化'''
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)
# 3. 图像展示
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,8),dpi=100)
axes[0].imshow(img,cmap=plt.cm.gray)
axes[0].set_title("原图")
axes[1].imshow(cl1,cmap=plt.cm.gray)
axes[1].set_title("自适应均衡化后的结果")
plt.show()

