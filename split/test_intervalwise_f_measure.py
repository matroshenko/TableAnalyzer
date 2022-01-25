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

if __name__ == '__main__':
    main()