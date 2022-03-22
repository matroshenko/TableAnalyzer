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
        tf.assert_equal(tf.shape(input)[0], 1)
        input = input[0]

        height = tf.shape(input)[0]
        width = tf.shape(input)[1]
        channels = tf.shape(input)[2]

        multiplier = tf.numpy_function( 
            self._create_reciprocal_cells_areas_matrix,
            [height, width, h_positions, v_positions], Tout=tf.float32)
        normalized_input = multiplier * input

        means = tf.zeros(shape=(tf.size(h_positions)+1, tf.size(v_positions)+1, channels))
        indices = tf.numpy_function(
            self._create_indices_matrix,
            [height, width, h_positions, v_positions], Tout=tf.int32)
        means = tf.tensor_scatter_nd_add(means, indices, normalized_input)
        
        if not self._keep_size:
            return tf.expand_dims(means, axis=0)

        result = tf.gather_nd(means, indices)
        result = tf.ensure_shape(result, shape=input.shape)
        return tf.expand_dims(result, axis=0)

    def _create_reciprocal_cells_areas_matrix(
            self, height, width, h_positions, v_positions):

        grid = GridStructure(
            [0] + list(h_positions) + [height], 
            [0] + list(v_positions) + [width])

        result = np.zeros(shape=(height, width, 1), dtype='float32')
        for i in range(grid.get_rows_count()):
            for j in range(grid.get_cols_count()):
                cell = grid.get_cell_rect(Rect(j, i, j+1, i+1))
                if cell.is_empty():
                    continue
                result[cell.top : cell.bottom, cell.left : cell.right] = 1 / cell.get_area()
        return result

    def _create_indices_matrix(
            self, height, width, h_positions, v_positions):
        
        grid = GridStructure(
            [0] + list(h_positions) + [height], 
            [0] + list(v_positions) + [width])

        result = np.zeros(shape=(height, width, 2), dtype='int32')

        for i in range(grid.get_rows_count()):
            for j in range(grid.get_cols_count()):
                cell = grid.get_cell_rect(Rect(j, i, j+1, i+1))
                result[cell.top : cell.bottom, cell.left : cell.right] = [i, j]
        return result
