from unittest import TestCase, main
from binarize_layer import BinarizeLayer
import tensorflow as tf

class BinarizeLayerTestCase(TestCase):
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

if __name__ == '__main__':
    main()