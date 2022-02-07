import tensorflow as tf
import tensorflow.keras as keras

from merge.grid_pooling_layer import GridPoolingLayer
from merge.concat_inputs_layer import ConcatInputsLayer
from merge.adjacency_f_measure import AdjacencyFMeasure
from datasets.ICDAR.markup_table import Table
from table.grid_structure import GridStructureBuilder
from table.cells_structure import CellsStructureBuilder
from utils.rect import Rect

class SharedFullyConvolutionalNetwork(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        # Original paper suggests to use kernel size = 7,
        # which leads to excessive memory consumption.
        self._conv1 = keras.layers.Conv2D(18, 3, padding='same', activation='relu')
        #self._pool1 = keras.layers.MaxPool2D()
        self._conv2 = keras.layers.Conv2D(18, 3, padding='same', activation='relu')
        self._conv3 = keras.layers.Conv2D(18, 3, padding='same', activation='relu')
        self._conv4 = keras.layers.Conv2D(18, 3, padding='same', activation='relu')
        #self._pool2 = keras.layers.MaxPool2D()

    def call(self, input):
        result = self._conv1(input)
        #result = self._pool1(result)
        result = self._conv2(result)
        result = self._conv3(result)
        result = self._conv4(result)
        #result = self._pool2(result)
        return result


class GridPoolingNetworkBlock(keras.layers.Layer):
    def __init__(self, should_output_predictions):
        super().__init__()
        self._should_output_predictions = should_output_predictions

        self._dilated_conv1 = keras.layers.Conv2D(6, 3, padding='same', activation='relu', dilation_rate=1)
        self._dilated_conv2 = keras.layers.Conv2D(6, 3, padding='same', activation='relu', dilation_rate=2)
        self._dilated_conv3 = keras.layers.Conv2D(6, 3, padding='same', activation='relu', dilation_rate=3)
        self._concat1 = keras.layers.Concatenate()

        self._upper_branch_conv = keras.layers.Conv2D(18, 1, activation='relu', name='upper_branch_conv')
        self._upper_branch_pool = GridPoolingLayer(True)
        self._lower_branch_conv = keras.layers.Conv2D(1, 1, activation='sigmoid', name='lower_branch_conv')
        self._lower_branch_pool = GridPoolingLayer(True)
        if should_output_predictions:
            self._prediction_layer = GridPoolingLayer(False)

        self._concat2 = keras.layers.Concatenate()

    def call(self, input, h_mask, v_mask):
        middle_result = self._concat1(
            [self._dilated_conv1(input), self._dilated_conv2(input), self._dilated_conv3(input)]
        )

        upper_result = self._upper_branch_conv(middle_result)
        upper_result = self._upper_branch_pool(upper_result, h_mask, v_mask)

        lower_result = self._lower_branch_conv(middle_result)
        if self._should_output_predictions:
            predictions = self._prediction_layer(lower_result, h_mask, v_mask)
            predictions = tf.squeeze(predictions, axis=3)
        lower_result = self._lower_branch_pool(lower_result, h_mask, v_mask)

        result = self._concat2([upper_result, middle_result, lower_result])
        if self._should_output_predictions:
            return [result, predictions]
        return result


class GridPoolingNetworkFinalBlock(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self._dilated_conv1 = keras.layers.Conv2D(6, 3, padding='same', activation='relu', dilation_rate=1)
        self._dilated_conv2 = keras.layers.Conv2D(6, 3, padding='same', activation='relu', dilation_rate=2)
        self._dilated_conv3 = keras.layers.Conv2D(6, 3, padding='same', activation='relu', dilation_rate=3)
        self._concat = keras.layers.Concatenate()
        self._conv1x1 = keras.layers.Conv2D(1, 1, activation='sigmoid')
        self._prediction_layer = GridPoolingLayer(False)

    def call(self, input, h_mask, v_mask):
        result = self._concat(
            [self._dilated_conv1(input), self._dilated_conv2(input), self._dilated_conv3(input)]
        )
        result = self._conv1x1(result)
        result = self._prediction_layer(result, h_mask, v_mask)
        result = tf.squeeze(result, axis=3)
        return result


class GridPoolingNetwork(keras.layers.Layer):
    def __init__(self, name):
        super().__init__(name=name)
        self._block1 = GridPoolingNetworkBlock(False)
        self._block2 = GridPoolingNetworkBlock(True)
        self._block3 = GridPoolingNetworkFinalBlock()

    def call(self, input, h_mask, v_mask):
        block1_output = self._block1(input, h_mask, v_mask)
        block2_output, probs1 = self._block2(block1_output, h_mask, v_mask)
        probs2 = self._block3(block2_output, h_mask, v_mask)
        return probs1, probs2


class CombineOutputsLayer(keras.layers.Layer):
    def call(self, up_prob, down_prob, left_prob, right_prob):
        merge_down_prob = (
            0.5 * up_prob[:, 1:, :] * down_prob[:, :-1, :] 
            + 0.25 * (up_prob[:, 1:, :] + down_prob[:, :-1, :])
        )
        merge_right_prob = (
            0.5 * left_prob[:, :, 1:] * right_prob[:, :, :-1]
            + 0.25 * (left_prob[:, :, 1:] + right_prob[:, :, :-1])
        )
        return merge_down_prob, merge_right_prob


class Model(keras.models.Model):
    def __init__(self):
        super().__init__()
        self._normalize_image_layer = keras.layers.experimental.preprocessing.Rescaling(
            scale=1./255)
        self._concat_inputs_layer = ConcatInputsLayer()
        self._sfcn = SharedFullyConvolutionalNetwork()
        self._up_branch = GridPoolingNetwork('up_branch')
        self._down_branch = GridPoolingNetwork('down_branch')
        self._left_branch = GridPoolingNetwork('left_branch')
        self._right_branch = GridPoolingNetwork('right_branch')
        self._combine_outputs1 = CombineOutputsLayer()
        self._combine_outputs2 = CombineOutputsLayer()

        self._metric = AdjacencyFMeasure()

        #self.output_names = [
        #    'merge_down_probs1', 'merge_right_probs1', 
        #    'merge_down_probs2', 'merge_right_probs2']

    def call(self, input_dict):
        image = input_dict['image']
        h_probs = input_dict['horz_split_points_probs']
        v_probs = input_dict['vert_split_points_probs']
        h_binary = input_dict['horz_split_points_binary']
        v_binary = input_dict['vert_split_points_binary']

        normalized_image = self._normalize_image_layer(image)
        input = self._concat_inputs_layer(normalized_image, h_probs, v_probs, h_binary, v_binary)
        sfcn_output = self._sfcn(input)
        
        up_prob1, up_prob2 = self._up_branch(sfcn_output, h_binary, v_binary)
        down_prob1, down_prob2 = self._down_branch(sfcn_output, h_binary, v_binary)
        left_prob1, left_prob2 = self._left_branch(sfcn_output, h_binary, v_binary)
        right_prob1, right_prob2 = self._right_branch(sfcn_output, h_binary, v_binary)

        merge_down_prob1, merge_right_prob1 = self._combine_outputs1(up_prob1, down_prob1, left_prob1, right_prob1)
        merge_down_prob2, merge_right_prob2 = self._combine_outputs2(up_prob2, down_prob2, left_prob2, right_prob2)

        return {
            'merge_down_probs1': merge_down_prob1,
            'merge_right_probs1': merge_right_prob1,
            'merge_down_probs2': merge_down_prob2,
            'merge_right_probs2': merge_right_prob2,
            'markup_table': None
        }

    def compute_metrics(self, input_dict, targets_dict, prediction, sample_weight):
        metric_results = super().compute_metrics(
            input_dict, targets_dict, prediction, sample_weight)

        markup_table = Table.from_tensor(tf.squeeze(targets_dict['markup_table'], axis=0))

        h_binary = tf.squeeze(input_dict['horz_split_points_binary'], axis=0).numpy()
        v_binary = tf.squeeze(input_dict['vert_split_points_binary'], axis=0).numpy()
        height = len(h_binary)
        width = len(v_binary)
        
        grid = GridStructureBuilder(Rect(0, 0, width, height), h_binary, v_binary).build()

        merge_down_mask = (tf.squeeze(prediction['merge_down_probs2'], axis=0) >= 0.5).numpy()
        merge_right_mask = (tf.squeeze(prediction['merge_right_probs2'], axis=0) >= 0.5).numpy()
        cells = CellsStructureBuilder(merge_right_mask, merge_down_mask).build()
        self._metric.update_state_eager(markup_table, grid, cells)
        
        return metric_results        
