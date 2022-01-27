from unittest import TestCase, main

import tensorflow as tf

import context
from split.model import Model

class ModelTestCase(TestCase):
    def test_output_shape(self):
        batch_size = 1
        height = 200
        width = 1000
        random_image = tf.random.uniform(shape=(batch_size, height, width, 3), minval=0, maxval=256, dtype='int32', seed=42)
        m = Model()
        outputs = m(random_image)

        self.assertEqual(
            outputs['horz_split_points_probs1'].shape, (batch_size, height))
        self.assertEqual(
            outputs['horz_split_points_probs2'].shape, (batch_size, height))
        self.assertEqual(
            outputs['horz_split_points_probs3'].shape, (batch_size, height,))
        self.assertEqual(
            outputs['horz_split_points_binary'].shape, (batch_size, height,))
        self.assertEqual(
            outputs['vert_split_points_probs1'].shape, (batch_size, width))
        self.assertEqual(
            outputs['vert_split_points_probs2'].shape, (batch_size, width))
        self.assertEqual(
            outputs['vert_split_points_probs3'].shape, (batch_size, width))
        self.assertEqual(
            outputs['vert_split_points_binary'].shape, (batch_size, width))

if __name__ == '__main__':
    main()