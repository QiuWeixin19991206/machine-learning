import numpy as np
'''视频读取'''
import cv2 as cv
cap = cv.VideoCapture('E:\c++\girl.mp4')#获取视频对象
# while(cap.isOpened()):#判断是否读取成功
#     ret, frame = cap.read()#获取每一帧图像
#     if ret == True:#获取成功显示图像
#         cv.imshow('frame', frame)
#     if cv.waitKey(100) & 0xff == ord('q'):
#         break
# cap.release()#释放视频对象
# cv.destroyAllWindows()
'''视频保存'''
# width = int(cap.get(3))#获取属性
# height = int(cap.get(4))
# out = cv.VideoWriter('E:\c++\out.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (width, height))
# while(cap.isOpened()):#判断是否读取成功
#     ret, frame = cap.read()#获取每一帧图像
#     if ret == True:#获取成功 保存
#         out.write(frame)
#     else:
#         break
# cap.release()#释放视频对象
# out.release()
# cv.destroyAllWindows()
'''视频追踪'''
# ret, frame = cap.read()# 2.获取第一帧图像，并指定目标位置
# r, h, c, w = 197, 141, 0, 208# 2.1 目标位置（行，高，列，宽）
# track_window = (c, r, w, h)#meanshift的初始搜索窗口
# roi = frame[r:r+h, c:c+w]# 2.2 指定目标的感兴趣区域
# hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)# 3. 感兴趣区域转HSV
# # 3.2 去除低亮度的值
# # mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
# roi_hist = cv.calcHist([hsv_roi], [0], None, [180], [0, 180])# 3.3 计算直方图
# cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)# 3.4 归一化
# term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)# 4. 目标追踪# 4.1 设置窗口搜索终止条件：最大迭代次数，窗口中心漂移最小值
# while(True):
#     ret, frame = cap.read()# 4.2 获取每一帧图像
#     if ret == True:
#         hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)# 4.3 计算直方图的反向投影
#         dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)#计算目标反向投影
#         ret, track_window = cv.meanShift(dst, track_window, term_crit)# 4.4 进行meanshift追踪
#         x, y, w, h = track_window# 4.5 将追踪的位置绘制在视频上，并进行显示
#         img2 = cv.rectangle(frame, (x, y), (x+w, y+h), 255, 2)
#         cv.imshow('frame', img2)
#         if cv.waitKey(60) & 0xFF == ord('a'):
#             break
#     else:
#         break
# cap.release()# 5. 资源释放
# cv.destroyAllWindows()
'''Camshift'''
ret, frame = cap.read()# 2.获取第一帧图像，并指定目标位置
r, h, c, w = 197, 141, 0, 208# 2.1 目标位置（行，高，列，宽）
track_window = (c, r, w, h)#meanshift的初始搜索窗口
roi = frame[r:r+h, c:c+w]# 2.2 指定目标的感兴趣区域
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)# 3. 感兴趣区域转HSV
# 3.2 去除低亮度的值
# mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv.calcHist([hsv_roi], [0], None, [180], [0, 180])# 3.3 计算直方图
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)# 3.4 归一化
term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)# 4. 目标追踪# 4.1 设置窗口搜索终止条件：最大迭代次数，窗口中心漂移最小值
while(True):
    ret, frame = cap.read()# 4.2 获取每一帧图像
    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)# 4.3 计算直方图的反向投影
        dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)#计算目标反向投影
        ret, track_window = cv.CamShift(dst, track_window, term_crit)# 4.4 进行meanshift追踪
        x, y, w, h = track_window# 4.5 将追踪的位置绘制在视频上，并进行显示
        pts = cv.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv.polylines(frame, [pts], True, 255, 2)
        cv.imshow('frame', img2)
        if cv.waitKey(60) & 0xFF == ord('a'):
            break
    else:
        break
cap.release()# 5. 资源释放
cv.destroyAllWindows()




