from unittest import TestCase, main

import tensorflow as tf
import numpy as np

import context
from merge.model import Model


class ModelTestCase(TestCase):
    def setUp(self):
        tf.config.set_visible_devices([], 'GPU')

    def test_output_shape(self):
        tf.random.set_seed(42)

        batch_size = 1
        height = 200
        width = 1000
        rows_count = 16
        cols_count = 32
        h_mask, h_positions = self._get_binary_vector_with_evenly_spaced_ones(height, rows_count-1)
        v_mask, v_positions = self._get_binary_vector_with_evenly_spaced_ones(width, cols_count-1)

        inputs = {
            'image': tf.random.uniform(shape=(batch_size, height, width, 3), minval=0, maxval=256, dtype='int32'),
            'horz_split_points_probs': tf.random.uniform(shape=(batch_size, height), dtype='float32'),
            'vert_split_points_probs': tf.random.uniform(shape=(batch_size, width), dtype='float32'),
            'horz_split_points_binary': tf.reshape(h_mask, shape=(batch_size, height)),
            'vert_split_points_binary': tf.reshape(v_mask, shape=(batch_size, width)),
            'horz_split_points_positions': h_positions,
            'vert_split_points_positions': v_positions
        }

        model = Model()
        outputs = model(inputs)

        expected_merge_down_shape = (batch_size, rows_count-1, cols_count)
        expected_merge_right_shape = (batch_size, rows_count, cols_count-1)

        self.assertEqual(
            outputs['merge_down_probs1'].shape, expected_merge_down_shape)
        self.assertEqual(
            outputs['merge_down_probs2'].shape, expected_merge_down_shape)
        self.assertEqual(
            outputs['merge_right_probs1'].shape, expected_merge_right_shape)
        self.assertEqual(
            outputs['merge_right_probs2'].shape, expected_merge_right_shape)

    def _get_binary_vector_with_evenly_spaced_ones(self, length, num_of_ones):
        result = np.zeros((length,), dtype='int32')
        space = (length - num_of_ones) // (num_of_ones + 1)
        assert space > 0
        index = space
        centers = []
        for i in range(num_of_ones):
            result[index] = 1
            centers.append(index)
            index += space + 1
        return tf.constant(result), tf.constant(centers)

if __name__ == '__main__':
    main(module='test_merge_model')