#特征提取 找角点 关键点
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
img = cv.imread('E:\c++\pantyhose.jpg')
'''harris'''
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)#转黑白图像
gray = np.float32(gray)#harris只支持float32
dst = cv.cornerHarris(gray, 2, 3, 0.04)#调用harris
img[dst>0.0001*dst.max()] = [0, 0, 255]#把角点划为红色
plt.figure(figsize=(10, 8), dpi=300)
plt.imshow(img[:, :, ::-1]), plt.title("jiaodian jiance")
plt.xticks([]),plt.yticks([])
plt.show()
'''shi-Tomasi'''
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# 2 角点检测
corners = cv.goodFeaturesToTrack(gray, 1000, 0.01, 10)#shi-Tomasi
# 3 绘制角点
for i in corners:
    x, y = i.ravel()
    x = np.int(x)
    y = np.int(y)
    cv.circle(img, (x, y), 5, (0, 0, 255), -1)
# 4 图像展示
plt.figure(figsize=(10,8),dpi=100)
plt.imshow(img[:,:,::-1]),plt.title('shi-tomasi角点检测')
plt.xticks([]), plt.yticks([])
plt.show()
'''SIFT算法'''
gray= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 2 sift关键点检测
# 2.1 实例化sift对象
sift = cv.SIFT_create()#更新了 旧：sift = cv.xfeatures2d.SIFT_create()
# 2.2 关键点检测：kp关键点信息包括方向，尺度，位置信息，des是关键点的描述符
kp,des = sift.detectAndCompute(gray,None)
# 2.3 在图像上绘制关键点的检测结果
cv.drawKeypoints(img,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# 3 图像显示
plt.figure(figsize=(15,10),dpi=300)
plt.imshow(img[:,:,::-1]),plt.title('sift检测')
plt.xticks([]), plt.yticks([])
plt.show()
'''FAST算法'''
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)#转黑白图像
fast = cv.FastFeatureDetector_create(threshold=20,nonmaxSuppression=True,type=cv.FAST_FEATURE_DETECTOR_TYPE_9_16)#实例化 获取FAST角点探测器
kp = fast.detect(img, None)#检测关键点
img2 = cv.drawKeypoints(img, kp, None, color=(0, 0, 255))

print("Threshold: {}".format(fast.getThreshold()))
print("nonmaxSuppression:{}".format(fast.getNonmaxSuppression()))
print("neighborhood:{}".format(fast.getType()))
print("Total Keypoints with nonmaxSuppression: {}".format(len(kp)))

fast.setNonmaxSuppression(0)#2.5关闭非极大值抑制
kp = fast.detect(img, None)
print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)))
#2.6绘制为进行非极大值抑制的结果
img3 = cv.drawKeypoints(img, kp, None, color=(0, 0, 255))
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8), dpi=100)
axes[0].imshow(img2[:, :, ::-1])
axes[0].set_title("jiaru加入非极大值抑制")
axes[1].imshow(img3[:, :, ::-1])
axes[1].set_title("weijia未加入非极大值抑制")
plt.show()
'''ORB算法'''
gray = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# orb = cv.Xfeatures2d.orb_create(nfeatures=5000)
orb = cv.ORB_create(nfeatures=5000)
kp, des = orb.detectAndCompute(gray, None)
img2 = cv.drawKeypoints(img, kp, None, color=(0, 0, 255))
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8), dpi=300)
axes[0].imshow(img[:, :, ::-1])
axes[0].set_title('photo1')
axes[1].imshow(img2[:, :, ::-1])
axes[1].set_title('photo2')
plt.show()
