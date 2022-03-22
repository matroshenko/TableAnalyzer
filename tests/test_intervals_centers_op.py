from unittest import TestCase, main

import tensorflow as tf
intervals_centers_module = tf.load_op_library('merge/ops/intervals_centers.so')

import context
from merge.model import Model


class IntervalsCentersOpTestCase(TestCase):
    def testSimple(self):
        input = tf.constant([0, 0, 1, 1, 1, 0, 1, 1])
        expected_output = tf.constant([3, 7])
        output = intervals_centers_module.intervals_centers(input)

        self.assertTrue(tf.reduce_all(expected_output == output))

if __name__ == '__main__':
    main()