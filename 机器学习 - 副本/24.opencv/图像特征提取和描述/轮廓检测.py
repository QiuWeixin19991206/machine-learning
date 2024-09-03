import cv2
import matplotlib.pyplot as plt
def cv_show(img,name) :
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread('img.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv_show(thresh, 'thresh')

binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
res = cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
cv_show(res, 'res')


# 阈值处理
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 检测轮廓
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 绘制轮廓
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

'''轮廓检测   轮廓近似  模板匹配'''

'''轮廓检测

目的：找到图像中的物体边界并返回这些边界的坐标。轮廓检测可以帮助识别和分析图像中的形状。

方法：轮廓检测通常在已经经过二值化处理的图像上进行。常见的步骤包括：

    图像二值化（如阈值处理或边缘检测）
    使用 cv2.findContours 函数检测轮廓

边缘检测

目的：检测图像中灰度值发生急剧变化的区域，即图像中的边缘。边缘通常对应于物体的边界。

方法：边缘检测通常使用一些滤波器或算子来识别灰度变化。常见的方法包括：

    Sobel算子：计算图像梯度，识别边缘。
    Canny算子：一个多阶段的边缘检测算法，效果更好，常用。
    Prewitt算子：类似于Sobel算子，但计算方法稍有不同。
    Laplacian算子：使用二阶导数来检测边缘。'''