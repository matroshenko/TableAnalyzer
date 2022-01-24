from enum import Enum

import tensorflow as tf
import tensorflow.keras as keras

class ProjectionDirection(Enum):
    Height = 0
    Width = 1

class ProjectionLayer(keras.layers.Layer):
    def __init__(self, direction, broadcast_to_original_shape):
        super().__init__()
        self._direction = direction
        self._broadcast_to_original_shape = broadcast_to_original_shape

    def call(self, input):
        num_of_dims = len(input.shape)
        assert num_of_dims in (3, 4)
        axis = num_of_dims - 2 if self._direction == ProjectionDirection.Height else num_of_dims - 3
        result = tf.reduce_mean(input, axis=axis, keepdims=True)
        if self._broadcast_to_original_shape:
            result = tf.broadcast_to(result, input.shape)
        return result