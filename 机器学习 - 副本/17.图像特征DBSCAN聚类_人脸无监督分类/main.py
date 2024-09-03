
import face_recognition
import pickle
import cv2
import os
import numpy as np
from sklearn.cluster import DBSCAN

dataset = 'dataset'
encodings = 'encodings.pickle'
detection_method = 'cnn'

'''读取到所有输入数据的路径'''
# 定义一个函数 list_files，接受一个参数 basePath，用于遍历指定路径下的所有文件
def list_files(basePath):
    # 使用 os.walk 函数遍历 basePath 下的所有目录和文件
    for (rootDir, dirName, filenames) in os.walk(basePath):
        # 遍历当前目录 rootDir 下的所有文件名 filenames
        for filename in filenames:
            # 使用 os.path.join 将当前目录路径 rootDir 和文件名 filename 合并为完整的文件路径 imagePath
            imagePath = os.path.join(rootDir, filename)
            # 使用 yield 关键字将生成器的下一个值设为当前的文件路径 imagePath
            yield imagePath

'''简化多个图像的组合和展示操作，使得图像处理流程更加高效和可控。'''
# 定义函数 build_montages，接受三个参数：image_list（图像列表）、image_shape（单个图像的形状）、montage_shape（蒙太奇的形状）
def build_montages(image_list, image_shape, montage_shape):
    # 初始化空列表，用于存储生成的蒙太奇图像
    image_montages = []

    # 创建一个空白图像数组，用于构建单个蒙太奇图像，其形状为（蒙太奇的总高度，蒙太奇的总宽度，3通道），数据类型为无符号8位整数（uint8）
    montage_image = np.zeros(shape=(image_shape[1] * (montage_shape[1]), image_shape[0] * montage_shape[0], 3),
                             dtype=np.uint8)

    # 初始的图像插入位置，初始为 [0, 0]
    cursor_pos = [0, 0]

    # 是否需要开始新的图像
    start_new_img = False

    # 遍历图像列表中的每个图像
    for img in image_list:
        # 标记不需要开始新的图像
        start_new_img = False

        # 调整图像大小为指定的 image_shape
        img = cv2.resize(img, image_shape)

        # 将调整大小后的图像 img 插入到 montage_image 的当前位置 cursor_pos 处
        montage_image[cursor_pos[1]:cursor_pos[1] + image_shape[1], cursor_pos[0]:cursor_pos[0] + image_shape[0]] = img

        # 更新 cursor_pos，使其指向下一个要插入图像的位置
        cursor_pos[0] += image_shape[0]

        # 如果当前行已经填满，移动到下一行
        if cursor_pos[0] >= montage_shape[0] * image_shape[0]:
            cursor_pos[1] += image_shape[1]
            cursor_pos[0] = 0

            # 如果当前列也已经填满，表示一个蒙太奇图像构建完成，将其添加到 image_montages 中
            if cursor_pos[1] >= montage_shape[1] * image_shape[1]:
                cursor_pos = [0, 0]
                image_montages.append(montage_image)

                # 重置 montage_image 以开始构建下一个蒙太奇图像
                montage_image = np.zeros(
                    shape=(image_shape[1] * (montage_shape[1]), image_shape[0] * montage_shape[0], 3),
                    dtype=np.uint8)
                start_new_img = True

    # 如果最后一个 montage_image 没有填满整个蒙太奇，确保将其添加到 image_montages 中
    if start_new_img is False:
        image_montages.append(montage_image)

    # 返回存储了所有生成蒙太奇图像的 image_montages 列表
    return image_montages



if __name__ == '__main__':
    imagePaths = list(list_files(dataset))
    '''使用神经网络 进行人脸检测和特征编码'''
    # data = []
    # for (i, imagePath) in enumerate(imagePaths):
    #     print('当前输入数据索引', i)
    #     # 读取到图像数据
    #     image = cv2.imread(imagePath)
    #     # 转换下顺序，因为一会要用工具包进行人脸检测，所以所必须得是固定格式
    #     rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     # 人脸检测
    #     boxes = face_recognition.face_locations(rgb, model=detection_method)
    #     # 向量编码
    #     encodings = face_recognition.face_encodings(rgb, boxes)
    #     # 组合得到的结果
    #     d = [{'imagePath': imagePath, 'loc': box, 'encoding': enc} for (box, enc) in zip(boxes, encodings)]
    #     data.extend(d) # 列表 data
    #     print()
    # # 保存到本地
    # f = open(encodings, 'wb')
    # f.write(pickle.dump(data))
    # f.close()

    # 读取保存好的向量
    data = pickle.loads(open(encodings, 'rb').read())
    data = np.array(data)
    encodings = [d['encoding'] for d in data]
    print(encodings[0])

    # ### 执行聚类操作
    dbscan = DBSCAN(metric='euclidean', n_jobs=-1) # 是一种经典的密度聚类算法，用于发现具有相似密度的样本点，并将它们归为同一类别
    dbscan.fit(encodings)

    print(dbscan.labels_)
    labelIDs = np.unique(dbscan.labels_)
    print(labelIDs) #array([-1,  0,  1,  2,  3,  4], dtype=int64) 说明有5个人
    labelIDs = np.array([0, 1, 2, 3, 4], dtype=np.int64)

    # 遍历每个聚类标签, 显示分类结果
    for labelID in labelIDs:
        # 找到属于当前标签的数据索引
        idxs = np.where(dbscan.labels_ == labelID)[0]
        # 随机选择部分数据索引，最多选择25个
        np.random.choice(idxs, size=min(25, len(idxs)))

        # 存储人脸图像的列表
        faces = []

        # 遍历当前标签下的每个数据索引
        for i in idxs:
            # 读取对应的图像路径
            image = cv2.imread(data[i]['imagePath'])
            # 提取人脸区域的坐标
            (top, right, bottom, left) = data[i]['loc']
            # 截取人脸图像并调整大小为(96, 96)
            face = image[top:bottom, left:right]
            face = cv2.resize(face, (96, 96))
            # 将处理后的人脸图像添加到列表中
            faces.append(face)

        # 构建蒙太奇图像，假设返回的是一个列表，取第一个元素作为蒙太奇图像
        montage = build_montages(faces, (96, 96), (5, 5))[0]

        # 显示蒙太奇图像
        cv2.imshow('res', montage)
        # 等待用户按下任意键后继续执行
        cv2.waitKey(0)

    print()
