import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
#读取图像 打印图像
# rain = cv.imread('rain.jpg', 1)
# plt.imshow(rain[:, :, ::-1])
# plt.show()
# view = cv.imread('view.jpg', 1)
# plt.imshow(view[:, :, ::-1])
# plt.show()
'''cv混合 两幅图大小需要相同'''
# imag1 = cv.add(rain, view)
# plt.imshow(imag1[:, :, ::-1])
# plt.show()
'''np混合'''
# imag2 = rain + view
# plt.imshow(imag2[:, :, ::-1])
# plt.show()
'''加权混合'''
# imag3 = cv.addWeighted(rain, 0.3, view, 0.7, 20)#一般权重和为1
# plt.imshow(imag3[:, :, ::-1])
# plt.show()
'''图像缩放pantyhose 读取路径不能有中文'''
# pantyhose = cv.imread('E:\c++\pantyhose.jpg', 1)
# plt.imshow(pantyhose[:, :, ::-1])
# plt.show()
# rows, cols = pantyhose.shape[:2]
# pantyhose2 = cv.resize(pantyhose, (2*cols, 2*rows))#绝对坐标放大
# plt.imshow(pantyhose2[:, :, ::-1])
# plt.show()
# pantyhose3 = cv.resize(pantyhose, None, fx=0.5, fy=0.5)#绝对坐标放大
# plt.imshow(pantyhose3[:, :, ::-1])
# plt.show()
'''平移图像'''
# pantyhose = cv.imread('E:\c++\pantyhose.jpg', 1)
# plt.imshow(pantyhose[:, :, ::-1])
# plt.show()
# rows, cols = pantyhose.shape[:2]
# M = np.float32([[1, 0, 100], [0, 1, 50]])
# dst = cv.warpAffine(pantyhose, M, (2*cols, 2*rows))
# plt.imshow(dst[:, :, ::-1])
# plt.show()
'''图像旋转'''
# pantyhose = cv.imread('E:\c++\pantyhose.jpg', 1)
# plt.imshow(pantyhose[:, :, ::-1])
# plt.show()
# rows, cols = pantyhose.shape[:2]
# M = cv.getRotationMatrix2D((cols/2, rows/2), 180, 1)#逆时针旋转
# res = cv.warpAffine(pantyhose, M, (cols, rows))
# plt.imshow(res[:, :, ::-1])
# plt.show()
'''仿射变换'''
# pantyhose = cv.imread('E:\c++\pantyhose.jpg', 1)
# rows, cols = pantyhose.shape[:2]
# pts1 = np.float32([[0, 0], [0, 1700], [1000, 1500]])
# pts2 = np.float32([[0, 0], [0, 1700], [1000, 1200]])
# M = cv.getAffineTransform(pts1, pts2)#逆时针旋转
# res = cv.warpAffine(pantyhose, M, (cols, rows))
# plt.imshow(res[:, :, ::-1])
# plt.show()
'''投射变换'''
pantyhose = cv.imread('E:\c++\pantyhose.jpg', 1)
rows, cols = pantyhose.shape[:2]
pts1 = np.float32([[0, 0], [0, 1700], [1000, 1500], [1000, 0]])
pts2 = np.float32([[0, 0], [0, 1600], [1000, 1200], [1000, 10]])
T = cv.getPerspectiveTransform(pts1, pts2)
dst = cv.warpPerspective(pantyhose, T, (cols, rows))
plt.imshow(dst[:, :, ::-1])
plt.show()
'''图像金字塔'''
# pantyhose = cv.imread('E:\c++\pantyhose.jpg', 1)
# up_img = cv.pyrUp(pantyhose)
# down_img = cv.pyrDown(pantyhose)
# cv.imshow('enlarge', up_img)
# cv.imshow('original', pantyhose)
# cv.imshow('shrink', down_img)
# cv.waitKey(0)
# cv.destroyAllWindows()




