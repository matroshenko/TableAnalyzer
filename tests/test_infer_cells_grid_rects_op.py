from unittest import TestCase, main

import tensorflow as tf

ops_module = tf.load_op_library('./merge/ops/ops.so')

class TestCellsStructure(TestCase):
    def test_simple(self):
        merge_right_mask = tf.constant([
            [True, True, False, False],
            [True, False, True, False],
            [False, False, False, False]
        ])
        merge_down_mask = tf.constant([
            [True, True, False, False, False],
            [False, False, False, False, False]
        ])
        expected_cells = tf.constant([
            [0, 0, 4, 2], 
            [4, 0, 5, 1], 
            [4, 1, 5, 2],
            [0, 2, 1, 3],
            [1, 2, 2, 3],
            [2, 2, 3, 3],
            [3, 2, 4, 3],
            [4, 2, 5, 3]
        ])
        cells = ops_module.infer_cells_grid_rects(merge_right_mask, merge_down_mask)
        self.assertTrue(tf.reduce_all(expected_cells == cells))


if __name__ == '__main__':
    main()