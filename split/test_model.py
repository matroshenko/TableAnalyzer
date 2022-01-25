from unittest import TestCase, main
import tensorflow as tf
from model import Model

class ModelTestCase(TestCase):
    def test_output_shape(self):
        batch_size = 1
        height = 200
        width = 1000
        random_image = tf.random.uniform(shape=(batch_size, height, width, 3), minval=0, maxval=256, dtype='int32', seed=42)
        m = Model()
        (horz_split_points_probs1, horz_split_points_probs2, horz_split_points_probs3,
            vert_split_points_probs1, vert_split_points_probs2, vert_split_points_probs3) = m(random_image)

        self.assertEqual(horz_split_points_probs1.shape, (batch_size, height))
        self.assertEqual(horz_split_points_probs2.shape, (batch_size, height))
        self.assertEqual(horz_split_points_probs3.shape, (batch_size, height,))
        self.assertEqual(vert_split_points_probs1.shape, (batch_size, width))
        self.assertEqual(vert_split_points_probs2.shape, (batch_size, width))
        self.assertEqual(vert_split_points_probs3.shape, (batch_size, width))

if __name__ == '__main__':
    main()