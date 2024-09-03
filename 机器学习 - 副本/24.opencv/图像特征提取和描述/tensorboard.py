# 导入所需的工具包
import numpy as np
import matplotlib.pyplot as plt

# tf中使用工具包
import tensorflow as tf
# 构建模型
from tensorflow.keras.models import Sequential
# 相关的网络层
from tensorflow.keras.layers import Dense,Dropout,Activation,BatchNormalization
# 导入辅助工具包
from tensorflow.keras import utils
# 正则化
from tensorflow.keras import regularizers
# 数据集
from tensorflow.keras.datasets import mnist


(x_train,y_train),(x_test,y_test) = mnist.load_data()

# 显示数据
plt.figure()
plt.imshow(x_train[1000],cmap="gray")
# 数据维度的调整
x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)
#%%
# 数据类型调整
x_train = x_train.astype('float32')
x_test = x_test.astype("float32")
#%%
# 归一化
x_train = x_train/255
x_test = x_test/255
#%%
# 将目标值转换成热编码的形式
y_train = utils.to_categorical(y_train,10)
y_test = utils.to_categorical(y_test,10)

# 使用序列模型进行构建
model = Sequential()
# 全连接层：2个隐层，一个输出层
# 第一个隐层:512个神经元，先激活后BN，随机失活
model.add(Dense(512,activation = "relu",input_shape=(784,)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
# 第二个隐层：512个神经元，先BN后激活，随机失活
model.add(Dense(512,kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.2))
# 输出层
model.add(Dense(10,activation="softmax"))

model.summary()

# 损失函数，优化器，评价指标
model.compile(loss= tf.keras.losses.categorical_crossentropy,optimizer = tf.keras.optimizers.Adam(),
              metrics=tf.keras.metrics.Accuracy())

#回调函数
tensorboad = tf.keras.callbacks.TensorBoard(log_dir='./graph')
#使用fit,指定训练集，epochs,batch_size,val,verbose
history = model.fit(x_train, y_train, epochs=4, validation_data=(x_test, y_test), batch_size=128,
                    verbose=1, callbacks=[tensorboad])
model.evaluate(x_test,y_test,verbose=1)
# 损失函数
plt.figure()
plt.plot(history.history['loss'],label="train")
plt.plot(history.history["val_loss"],label="val")
plt.legend()
plt.grid()

# 准确率
plt.figure()
plt.plot(history.history['accuracy'],label="train")
plt.plot(history.history["val_accuracy"],label="val")
plt.legend()
plt.grid()

# 保存
model.save("model.h5")
# 记载
loadmodel = tf.keras.models.load_model("model.h5")
loadmodel.evaluate(x_test,y_test,verbose=1)

#

# with summary_writer.as_default():
#     tf.summary.scalar('d_loss', float(d_loss), step=epoch)
#     tf.summary.scalar('g_loss', float(g_loss), step=epoch)
#     img1 = generate_big_image(fake_img)
#     tf.summary.image("fake_image", img1, step=epoch)
#     img2 = generate_big_image(batch_xy)
#     tf.summary.image("real_image", img2, step=epoch)