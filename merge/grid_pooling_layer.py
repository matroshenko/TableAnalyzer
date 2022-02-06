import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

from table.grid_structure import GridStructureBuilder
from utils.rect import Rect


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
        channels = input.shape[3]

        grid = GridStructureBuilder(Rect(0, 0, width, height), h_mask_array, v_mask_array).build()

        input = tf.reshape(input, shape=(height*width, channels))
        multiplier = self._create_multiplier_vector(grid)
        normalized_input = multiplier * input

        means = tf.zeros(shape=(grid.get_rows_count() * grid.get_cols_count(), channels))
        indices = self._create_indices(grid)
        means = tf.tensor_scatter_nd_add(means, np.expand_dims(indices, 1), normalized_input)
        
        if not self._keep_size:
            return tf.reshape(means, shape=(1, grid.get_rows_count(), grid.get_cols_count(), channels))

        result = tf.gather(means, indices)
        return tf.reshape(result, shape=(1, height, width, channels))

    def _create_multiplier_vector(self, grid):
        height = grid.get_bounding_rect().get_height()
        width = grid.get_bounding_rect().get_width()

        result = np.zeros(shape=(height * width, 1), dtype='float32')
        for i in range(grid.get_rows_count()):
            for j in range(grid.get_cols_count()):
                cell = grid.get_cell_rect(Rect(j, i, j+1, i+1))
                self._update_vector(grid, cell, result, 1 / cell.get_area())
        return result

    def _update_vector(self, grid, cell, vector, value):
        width = grid.get_bounding_rect().get_width()
        for i in range(cell.top, cell.bottom):
            for j in range(cell.left, cell.right):
                vector[i * width + j] = value

    def _create_indices(self, grid):
        height = grid.get_bounding_rect().get_height()
        width = grid.get_bounding_rect().get_width()
        result = np.zeros(shape=(height * width,), dtype='int32')

        cols_count = grid.get_cols_count()
        for i in range(grid.get_rows_count()):
            for j in range(cols_count):
                cell = grid.get_cell_rect(Rect(j, i, j+1, i+1))
                self._update_vector(grid, cell, result, i * cols_count + j)
        return result
