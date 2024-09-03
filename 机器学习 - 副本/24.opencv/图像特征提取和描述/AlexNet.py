import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from tensorflow import keras
import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)
np.random.seed(1234)
assert tf.__version__.startswith('2.')

class Mynet(keras.Model):
    def __init__(self):
        super(Mynet, self).__init__()

        self.net = keras.Sequential([
            layers.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
            layers.MaxPool2D(pool_size=3, strides=2),
            layers.Conv2D(256, kernel_size=11, padding='same', activation='relu'),
            layers.MaxPool2D(pool_size=3, strides=2),
            layers.Conv2D(384, kernel_size=11, padding='same', activation='relu'),
            layers.Conv2D(384, kernel_size=11, padding='same', activation='relu'),
            layers.Conv2D(256, kernel_size=11, padding='same', activation='relu'),
            layers.MaxPool2D(pool_size=3, strides=2),
            layers.Flatten(),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])


    def call(self, inputs, training=None, mask=None):
        return self.net(inputs)

def main():
    (train_images, train_label), (test_images, test_labels) = datasets.mnist.load_data()
    # %%
    # 维度调整
    train_images = np.reshape(train_images, (train_images.shape[0], train_images.shape[1], train_images.shape[2], 1))
    test_images = np.reshape(test_images, (test_images.shape[0], test_images.shape[1], test_images.shape[2], 1))
    print('train_images, test_images', train_images.shape, test_images.shape)
    # %%
    # 对训练数据进行抽样
    def get_train(size):
        # 随机生成index
        index = np.random.randint(0, train_images.shape[0], size)
        # 选择图像并进行resize
        resized_image = tf.image.resize_with_pad(train_images[index], 227, 227)
        return resized_image.numpy(), train_label[index]

    # %%
    # 对测试数据进行抽样
    def get_test(size):
        # 随机生成index
        index = np.random.randint(0, test_images.shape[0], size)
        # 选择图像并进行resize
        resized_image = tf.image.resize_with_pad(test_images[index], 227, 227)
        return resized_image.numpy(), test_labels[index]

    # %%
    # 抽样结果
    train_images, train_label = get_train(256)
    test_images, test_labels = get_test(128)
    print('train_images{} test_images{}, train_label{}, test_labels{}'.format(train_images.shape, test_images.shape, train_label.shape, test_labels.shape))

    plt.imshow(train_images[4].astype(np.int8).squeeze(), cmap='gray')
    plt.show()

    model = Mynet()

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                loss=tf.keras.losses.sparse_categorical_crossentropy
                , metrics=['accuracy'])
    model.fit(train_images,train_label,batch_size=128,epochs=3,validation_split=0.1,verbose=1)
    model.evaluate(test_images,test_labels,verbose=1)



if __name__ == '__main__':
    main()