import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

from datasets.ICDAR.grid_structure import GridStructure
from datasets.ICDAR.rect import Rect


class GridPoolingLayer(keras.layers.Layer):
    def __init__(self, keep_size):
        super().__init__()
        self._keep_size = keep_size

    def call(self, input, h_positions, v_positions):
        assert tf.executing_eagerly()
        assert input.shape[0] == 1
        input_array = tf.squeeze(input, axis=0).numpy()
        h_positions_list = tf.squeeze(h_positions, axis=0).numpy().tolist()
        v_positions_list = tf.squeeze(v_positions, axis=0).numpy().tolist()

        height = input_array.shape[0]
        width = input_array.shape[1]
        channels = input_array.shape[2]

        grid = GridStructure(
            [0] + h_positions_list + [height], 
            [0] + v_positions_list + [width])
        
        result_shape = (
            input.shape if self._keep_size 
            else (1, grid.get_rows_count(), grid.get_cols_count(), channels))
        result = np.zeros(result_shape, dtype='float32')

        for i in range(grid.get_rows_count()):
            for j in range(grid.get_cols_count()):
                cell = grid.get_cell_rect(Rect(j, i, j+1, i+1))
                averaged_block = self._get_averaged_block(input_array, cell)
                if self._keep_size:
                    result[0, cell.top : cell.bottom, cell.left : cell.right, :] = averaged_block
                else:
                    result[0, i, j, :] = averaged_block
        
        return tf.convert_to_tensor(result)


    def _get_averaged_block(self, input, cell):
        block = input[cell.top : cell.bottom, cell.left : cell.right, :]
        averaged_block = np.mean(block, axis=(0, 1), keepdims=True)
        if self._keep_size:
            averaged_block = np.broadcast_to(averaged_block, block.shape)
        return averaged_block
