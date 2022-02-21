from unittest import TestCase, main

import tensorflow as tf
import numpy as np

import context
from merge.model import Model


class ModelTestCase(TestCase):
    def test_output_shape(self):
        tf.random.set_seed(42)

        batch_size = 1
        height = 200
        width = 1000
        rows_count = 16
        cols_count = 32
        h_mask = self._get_binary_vector_with_evenly_spaced_ones(height, rows_count-1)
        v_mask = self._get_binary_vector_with_evenly_spaced_ones(width, cols_count-1)

        inputs = {
            'image': tf.random.uniform(shape=(batch_size, height, width, 3), minval=0, maxval=256, dtype='int32'),
            'horz_split_points_probs': tf.random.uniform(shape=(batch_size, height), dtype='float32'),
            'vert_split_points_probs': tf.random.uniform(shape=(batch_size, width), dtype='float32'),
            'horz_split_points_binary': tf.reshape(h_mask, shape=(batch_size, height)),
            'vert_split_points_binary': tf.reshape(v_mask, shape=(batch_size, width))
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
        for i in range(num_of_ones):
            result[index] = 1
            index += space + 1
        return tf.constant(result)

if __name__ == '__main__':
    main(module='test_merge_model')