import cv2
import numpy as np

def cv_show(img,name=None) :
    cv2.imshow(name,img)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

template = cv2.imread('mobanjiequ.jpg', 0)
img_rgb = cv2.imread('mobanpipei.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
# cv_show(template)
# cv_show(img_rgb)
# cv_show(img_gray)

h, w = template.shape[:2]
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
#取匹配程度大于%80的坐标
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):#*号表示可选参数
    bottom_right =(pt[0] + w, pt[1] + h)
    cv2.rectangle(img_rgb, pt, bottom_right, (0, 0, 255), 2)
cv2.imshow ('img_rgb', img_rgb)
cv2.waitKey(0)











