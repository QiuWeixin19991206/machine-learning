import numpy as np
import cv2
import matplotlib.pyplot as plt
#创建图像
img = np.zeros((511, 511, 3), np.uint8)
#绘制图形
cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5)
cv2.circle(img, (256, 256), 60, (0, 0, 255), -1)#线条大小=-1填充
cv2.putText(img, "hello", (100, 150), cv2.FONT_HERSHEY_COMPLEX, 5, (255, 255, 255), 3)
#显示结果
plt.imshow(img[:, :, ::-1])
plt.show()
