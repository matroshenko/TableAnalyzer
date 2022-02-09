import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

from table.grid_structure import GridStructure
from utils.rect import Rect


class GridPoolingLayer(keras.layers.Layer):
    def __init__(self, keep_size):
        super().__init__()
        self._keep_size = keep_size

    def call(self, input, h_positions, v_positions):
        assert tf.executing_eagerly()
        assert input.shape[0] == 1

        height = input.shape[1]
        width = input.shape[2]
        channels = input.shape[3]

        grid = GridStructure([0] + h_positions + [height], [0] + v_positions + [width])

        input = tf.squeeze(input, axis=0)
        multiplier = self._create_reciprocal_cells_areas_matrix(grid)
        normalized_input = multiplier * input

        means = tf.zeros(shape=(grid.get_rows_count(), grid.get_cols_count(), channels))
        indices = self._create_indices_matrix(grid)
        means = tf.tensor_scatter_nd_add(means, indices, normalized_input)
        
        if not self._keep_size:
            return tf.expand_dims(means, axis=0)

        result = tf.gather_nd(means, indices)
        return tf.expand_dims(result, axis=0)

    def _create_reciprocal_cells_areas_matrix(self, grid):
        height = grid.get_bounding_rect().get_height()
        width = grid.get_bounding_rect().get_width()

        result = np.zeros(shape=(height, width, 1), dtype='float32')
        for i in range(grid.get_rows_count()):
            for j in range(grid.get_cols_count()):
                cell = grid.get_cell_rect(Rect(j, i, j+1, i+1))
                if cell.is_empty():
                    continue
                result[cell.top : cell.bottom, cell.left : cell.right] = 1 / cell.get_area()
        return result

    def _create_indices_matrix(self, grid):
        height = grid.get_bounding_rect().get_height()
        width = grid.get_bounding_rect().get_width()
        result = np.zeros(shape=(height, width, 2), dtype='int32')

        for i in range(grid.get_rows_count()):
            for j in range(grid.get_cols_count()):
                cell = grid.get_cell_rect(Rect(j, i, j+1, i+1))
                result[cell.top : cell.bottom, cell.left : cell.right] = [i, j]
        return result
