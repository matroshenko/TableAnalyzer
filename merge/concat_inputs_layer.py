import tensorflow as tf
import tensorflow.keras as keras


class ConcatInputsLayer(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, normalized_image, h_probs, v_probs, 
            h_binary, v_binary, h_positions, v_positions):

        tf.debugging.assert_shapes([
            (normalized_image, (1, 'H', 'W', 3)),
            (h_probs, (1, 'H')),
            (v_probs, (1, 'W')),
            (h_binary, (1, 'H')),
            (v_binary, (1, 'W'))
        ])    
        height = tf.shape(normalized_image)[1]
        width = tf.shape(normalized_image)[2]

        broadcasted_h_probs = self._broadcast_horz_mask(h_probs, height, width)
        broadcasted_v_probs = self._broadcast_vert_mask(v_probs, height, width)
        broadcasted_h_binary = self._broadcast_horz_mask(h_binary, height, width)
        broadcasted_v_binary = self._broadcast_vert_mask(v_binary, height, width)
        grid_image = self._create_grid_image(h_positions, v_positions, height, width)

        return tf.concat([
            normalized_image, 
            broadcasted_h_probs, 
            broadcasted_v_probs, 
            tf.cast(broadcasted_h_binary, tf.float32),
            tf.cast(broadcasted_v_binary, tf.float32),
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

    def _create_grid_image(self, h_positions, v_positions, height, width):
        result = tf.zeros(shape=(height, width))
        result = self._draw_horz_lines(result, h_positions)
        result = tf.transpose(result)
        result = self._draw_horz_lines(result, v_positions)
        return tf.transpose(result)

    def _draw_horz_lines(self, image, positions):
        updates = tf.ones(shape=(tf.size(positions), tf.shape(image)[1]))
        return tf.tensor_scatter_nd_update(image, positions, updates)