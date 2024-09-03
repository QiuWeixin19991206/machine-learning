import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
img = cv.imread('E:\c++\mobanpipei.jpg')
'''模板检测'''
template = cv.imread('E:\c++\mobanjiequ.jpg')
h, w, l = template.shape
res = cv.matchTemplate(img, template, method=cv.TM_CCORR)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv.rectangle(img, top_left, bottom_right, (0, 255, 0), 10)
plt.imshow(img[:, :, ::-1])
plt.title("out")
plt.xticks([])
plt.yticks([])
plt.show()
'''霍夫线检测'''
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# edges = cv.Canny(gray, 100, 200)
# lines = cv.HoughLines(edges, 0.8, np.pi/180, 150)
# for line in lines:
#     rho, theta = line[0]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a * rho
#     y0 = b * rho
#     x1 = int(x0 + 1000 * (-b))
#     y1 = int(y0 + 1000 * (a))
#     x2 = int(x0 - 1000 * (-b))
#     y2 = int(y0 - 1000 * (a))
#     cv.line(img, (x1, y1), (x2, y2), (0, 256, 0), 10)
# plt.figure(figsize=(10, 8), dpi=100)
# plt.imshow(img[:, :, ::-1])
# plt.title('huofujiance')
# plt.xticks([]),plt.yticks([])
# plt.show()
'''霍夫圆检测'''
# planets = cv.imread('E:\c++\mobanpipei.jpg')
# gay_img = cv.cvtColor(planets, cv.COLOR_BGRA2GRAY)
# # 2 进行中值模糊，去噪点
# img = cv.medianBlur(gay_img, 7)
# # 3 霍夫圆检测
# circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 200, param1=100, param2=50, minRadius=0, maxRadius=100)
# # 4 将检测结果绘制在图像上
# for i in circles[0, :]:  # 遍历矩阵每一行的数据
#     # 绘制圆形
#     cv.circle(planets, (i[0], i[1]), i[2], (0, 255, 0), 2)
#     # 绘制圆心
#     cv.circle(planets, (i[0], i[1]), 2, (0, 0, 255), -1)
# # 5 图像显示
# plt.figure(figsize=(10,8),dpi=100)
# plt.imshow(planets[:,:,::-1]),plt.title('霍夫变换圆检测')
# plt.xticks([]), plt.yticks([])
# plt.show()





