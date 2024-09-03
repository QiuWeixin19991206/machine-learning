import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

'''tf.image进行增强'''
cat = plt.imread(r'F:\qwx\学习计算机视觉\机器学习\黑马\pantyhose.jpg')
cat1 = tf.image.random_flip_left_right(cat)#左右翻转
cat2 = tf.image.random_flip_up_down(cat)#上下反转
cat3 = tf.image.random_crop(cat, (1750, 1000, 3))#裁剪
cat4 = tf.image.random_brightness(cat, 0.5)#调节亮度
cat5 = tf.image.random_hue(cat, 0.5)#调节色调

fig, axes = plt.subplots(nrows=1, ncols=6,
                         figsize=(10, 8), dpi=300)
axes[0].imshow(cat)
axes[0].set_title("photo")
axes[1].imshow(cat1)
axes[1].set_title("left_right")
axes[2].imshow(cat2)
axes[2].set_title("up_down")
axes[3].imshow(cat3)
axes[3].set_title("crop")
axes[4].imshow(cat4)
axes[4].set_title("brightness")
axes[5].imshow(cat5)
axes[5].set_title("hue")
plt.show()




