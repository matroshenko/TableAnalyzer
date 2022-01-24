from unittest import TestCase, main
from binarize_layer import BinarizeLayer
import tensorflow as tf
import time

class BinarizeLayerTestCase(TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print('%s: %.3f' % (self.id(), t))

    def test_one_pixel(self):
        input = tf.constant([0.9, 0.1, 0.9])
        expected_output = tf.constant([1, 1, 1])
        output = BinarizeLayer(0.75)(input)
        self.assertTrue(all(expected_output == output))

    def test_two_pixels(self):
        input = tf.constant([0.9, 0.1, 0.1, 0.9])
        expected_output = tf.constant([1, 0, 0, 1])
        output = BinarizeLayer(0.75)(input)
        self.assertTrue(all(expected_output == output))

    def test_large_input(self):
        input = tf.random.uniform(shape=(1000,), minval=0, maxval=1, seed=42)
        output = BinarizeLayer(0)(input)
        expected_output = tf.cast(input >= 0.5, output.dtype)
        self.assertTrue(all(expected_output == output))


if __name__ == '__main__':
    main(module='test_binarize_layer')