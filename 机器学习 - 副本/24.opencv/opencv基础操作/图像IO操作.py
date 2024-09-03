'''图像的IO操作，读取和保存方法
在图像上绘制几何图形
怎么获取图像的属性
怎么访问图像的像素，进行通道分离，合并等
怎么实现颜色空间的变换
图像的算术运算'''
import cv2
import matplotlib.pyplot as plt
import numpy as np

print(cv2.__version__)#读取版本
lena = cv2.imread('1.png', 1)#读取图像 1彩色0灰度-1包括alpha通道
cv2.imshow("image", lena)#显示图像
cv2.waitKey(0)#等待窗口
cv2.destroyAllWindows()
#matplotlib 是反转的 所以：：-1
plt.imshow(lena[:, :, ::-1])
plt.show()
#保存图像
cv2.imwrite("girlfriend.png", lena)




