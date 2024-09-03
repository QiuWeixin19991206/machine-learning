import keras
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers

if __name__ == '__main__':

    # 方法1
    # model = keras.Sequential([
    #     layers.Dense(3, activation='relu', kernel_initializer='he_normal', name='layer1', input_shape=(3,)),
    #     layers.Dense(2, activation='relu', kernel_initializer='he_normal', name='layer2'),
    #     layers.Dense(2, activation='sigmoid', kernel_initializer='he_normal', name='layer3')
    # ],name = 'my_Sequential')
    # model.summary()
    # 方法2 function创建
    # 方法3 model创建
    class MyModel(keras.Model):

        def __init__(self):
            super(MyModel, self).__init__()
            self.layer1 = layers.Dense(3, activation='relu', kernel_initializer='he_normal', name='layer1', input_shape=(3,))
            self.layer2 = layers.Dense(2, activation='relu', kernel_initializer='he_normal', name='layer2')
            self.layer3 =layers.Dense(2, activation='sigmoid', kernel_initializer='he_normal', name='layer3')

        def call(self, inputs, training=None, mask=None):
            x = self.layer1(inputs)
            x = self.layer2(x)
            x = self.layer3(x)
            return x

    model = MyModel()
    x = tf.ones((1, 3))
    y = model(x)
    print(y)