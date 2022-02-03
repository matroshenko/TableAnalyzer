import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

from datasets.ICDAR.grid_structure import GridStructureBuilder
from datasets.ICDAR.rect import Rect


class GridPoolingLayer(keras.layers.Layer):
    def __init__(self, keep_size):
        super().__init__()
        self._keep_size = keep_size

    def call(self, input, h_mask, v_mask):
        assert tf.executing_eagerly()
        assert input.shape[0] == 1

        h_mask_array = tf.squeeze(h_mask, axis=0).numpy()
        v_mask_array = tf.squeeze(v_mask, axis=0).numpy()

        height = input.shape[1]
        width = input.shape[2]

        grid = GridStructureBuilder(Rect(0, 0, width, height), h_mask_array, v_mask_array).build()

        rows = []
        for i in range(grid.get_rows_count()):
            blocks = []
            for j in range(grid.get_cols_count()):
                cell = grid.get_cell_rect(Rect(j, i, j+1, i+1))
                averaged_block = self._get_averaged_block(input, cell)
                blocks.append(averaged_block)
            row = tf.concat(blocks, axis=2)
            rows.append(row)

        return tf.concat(rows, axis=1)


    def _get_averaged_block(self, input, cell):
        block = input[:, cell.top : cell.bottom, cell.left : cell.right, :]
        averaged_block = tf.reduce_mean(block, axis=(1, 2), keepdims=True)
        if self._keep_size:
            averaged_block = np.broadcast_to(averaged_block, block.shape)
        return averaged_block
