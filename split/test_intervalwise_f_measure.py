from unittest import TestCase, main
from intervalwise_f_measure import IntervalwiseFMeasure
import tensorflow as tf

class IntervalwiseFMeasureTestCase(TestCase):
    def test_simple(self):
        markup_mask = tf.constant([[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]])
        predicted_mask = tf.constant([[0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1]])
        metric = IntervalwiseFMeasure()
        metric.update_state(markup_mask, predicted_mask)

        self.assertEqual(metric.result(), 2/3)

    def test_large_input(self):
        shape = (1, 100)
        tf.random.set_seed(42)
        markup_mask = tf.random.uniform(shape, 0, 2, dtype='int32')
        predicted_mask = tf.random.uniform(shape, 0, 2, dtype='int32')
        metric = IntervalwiseFMeasure()
        metric.update_state(markup_mask, predicted_mask)

if __name__ == '__main__':
    main()