import tensorflow as tf
import tensorflow.keras as keras


class SharedFullyConvolutionalNetwork(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        # Original paper suggests to use kernel size = 7,
        # which leads to excessive memory consumption.
        self._conv1 = keras.layers.Conv2D(18, 3, padding='same', activation='relu')
        self._pool1 = keras.layers.MaxPool2D()
        self._conv2 = keras.layers.Conv2D(18, 3, padding='same', activation='relu')
        self._conv3 = keras.layers.Conv2D(18, 3, padding='same', activation='relu')
        self._conv4 = keras.layers.Conv2D(18, 3, padding='same', activation='relu')
        self._pool2 = keras.layers.MaxPool2D()

    def call(self, input):
        result = self._conv1(input)
        result = self._pool1(result)
        result = self._conv2(result)
        result = self._conv3(result)
        result = self._conv4(result)
        result = self._pool2(result)
        return result