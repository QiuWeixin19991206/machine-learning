
'''
傅里叶变换的作用
·高频:变化剧烈的灰度分量，例如边界·
·低频:变化缓慢的灰度分量，例如一片大海

滤波
低通滤波器:只保留低频，会使得图像模糊·
高通滤波器:只保留高频，会使得图像细节增强
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取灰度图像
img = cv2.imread('img.png', 0)
if img is None:
    raise FileNotFoundError('img.jpg not found')

# 将图像转换为浮点型
img_float32 = np.float32(img)

# 进行DFT
dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)

# 进行频谱中心化
dft_shift = np.fft.fftshift(dft)

# 计算幅度谱
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)  # 加1防止log(0)

# 显示原图和幅度谱
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

'''低通 高通   滤波器'''
import numpy as np
import cv2
from matplotlib import pyplot as plt
'''低通'''
img = cv2.imread('img.png', 0)
img_float32 = np.float32(img)

# 傅里叶变换
dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

rows, cols = img.shape
crow, ccol = int(rows / 2), int(cols / 2)

# 创建低通滤波器
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

# 应用滤波器
fshift = dft_shift * mask
f_ishift = np.fft.ifftshift(fshift)

# 逆傅里叶变换
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

# 显示原始图像和处理后的结果
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('Result'), plt.xticks([]), plt.yticks([])
plt.show()

'''高通'''

img = cv2.imread('img.png', 0)
img_float32 = np.float32(img)

# 傅里叶变换
dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

rows, cols = img.shape
crow, ccol = int(rows / 2), int(cols / 2)

# 创建高通滤波器
ask_= np.ones((rows,cols,2), np.uint8)
mask[crow-30 : crow+30, ccol-30:ccol+30]=0

# 应用滤波器
fshift = dft_shift * mask
f_ishift = np.fft.ifftshift(fshift)

# 逆傅里叶变换
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

plt.subplot(121), plt.imshow(img, cmap= 'gray')
plt.title('Input Image2'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap = 'gray')
plt.title('Result2'), plt.xticks([]), plt.yticks([])
plt.show()








