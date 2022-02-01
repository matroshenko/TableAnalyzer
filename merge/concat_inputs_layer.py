import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

from utils import get_intervals_of_ones


class ConcatInputsLayer(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, table_image, h_probs, v_probs, h_binary, v_binary):
        assert tf.executing_eagerly()
        height = table_image.shape[1]
        width = table_image.shape[2]
        assert h_probs.shape[1] == height
        assert v_probs.shape[1] == width
        assert h_binary.shape == h_probs.shape
        assert v_binary.shape == v_probs.shape

        broadcasted_h_probs = self._broadcast_horz_mask(h_probs, height, width)
        broadcasted_v_probs = self._broadcast_vert_mask(v_probs, height, width)
        broadcasted_h_binary = self._broadcast_horz_mask(h_binary, height, width)
        broadcasted_v_binary = self._broadcast_vert_mask(v_binary, height, width)
        grid_image = self._create_grid_image(h_binary, v_binary, height, width)

        return tf.concat([
            table_image, 
            broadcasted_h_probs, 
            broadcasted_v_probs, 
            broadcasted_h_binary,
            broadcasted_v_binary,
            grid_image
        ], axis=3)

    def _broadcast_horz_mask(self, mask, height, width):
        mask = tf.expand_dims(mask, 2)
        mask = tf.expand_dims(mask, 3)
        return tf.broadcast_to(mask, (1, height, width, 1))

    def _broadcast_vert_mask(self, mask, height, width):
        mask = tf.expand_dims(mask, 1)
        mask = tf.expand_dims(mask, 3)
        return tf.broadcast_to(mask, (1, height, width, 1))

    def _create_grid_image(self, h_binary, v_binary, height, width):
        h_binary = tf.squeeze(h_binary).numpy()
        v_binary = tf.squeeze(v_binary).numpy()
        horz_split_points_intervals = get_intervals_of_ones(h_binary)
        vert_split_points_intervals = get_intervals_of_ones(v_binary)

        result = np.zeros(shape=(1, height, width, 1), dtype='int32')
        # Original paper recommends to draw lines with thickness=7,
        # but we think, that thickness=1 is enough.
        for h_interval in horz_split_points_intervals:
            x = h_interval.get_center()
            result[0, x, :, 0] = 1
        for v_interval in vert_split_points_intervals:
            y = v_interval.get_center()
            result[0, :, y, 0] = 1

        return tf.constant(result)