from unittest import TestCase, main

import tensorflow as tf

import context
from merge.grid_pooling_layer import GridPoolingLayer


class GridPoolingLayerTestCase(TestCase):
    def setUp(self):
        self.input = tf.constant([[
            [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
            [[10, 11], [12, 13], [14, 15], [16, 17], [18, 19]],
            [[20, 21], [22, 23], [24, 25], [26, 27], [28, 29]],
            [[30, 31], [32, 33], [34, 35], [36, 37], [38, 39]],
        ]], dtype='float32')
        self.h_positions = tf.constant([[1, 3]], dtype='int32')
        self.v_positions = tf.constant([[2, 3]], dtype='int32')

    def test_pool_no_keep_size(self):
        expected_output = tf.constant([[
            [[1, 2], [4, 5], [7, 8]],
            [[16, 17], [19, 20], [22, 23]],
            [[31, 32], [34, 35], [37, 38]]
        ]], dtype = 'float32')
        layer = GridPoolingLayer(False)
        output = layer(self.input, self.h_positions, self.v_positions)
        self.assertTrue(tf.reduce_all(expected_output == output))

    def test_pool_keep_size(self):
        expected_output = tf.constant([[
            [[1, 2], [1, 2], [4, 5], [7, 8], [7, 8]],
            [[16, 17], [16, 17], [19, 20], [22, 23], [22, 23]],
            [[16, 17], [16, 17], [19, 20], [22, 23], [22, 23]],
            [[31, 32], [31, 32], [34, 35], [37, 38], [37, 38]]
        ]], dtype = 'float32')
        layer = GridPoolingLayer(True)
        output = layer(self.input, self.h_positions, self.v_positions)
        self.assertTrue(tf.reduce_all(expected_output == output))

if __name__ == '__main__':
    main()