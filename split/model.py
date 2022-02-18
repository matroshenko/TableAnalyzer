import tensorflow as tf
import tensorflow.keras as keras

from split.projection_layer import ProjectionLayer, ProjectionDirection
from split.binarize_layer import BinarizeLayer
from table.markup_table import Table
from table.grid_structure import GridStructureBuilder
from utils.rect import Rect
from metrics.adjacency_f_measure import AdjacencyFMeasure


class SharedFullyConvolutionalNetwork(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self._conv1 = keras.layers.Conv2D(18, 7, padding='same', activation='relu')
        self._conv2 = keras.layers.Conv2D(18, 7, padding='same', activation='relu')
        self._conv3 = keras.layers.Conv2D(18, 7, padding='same', activation='relu', dilation_rate=2)

    def call(self, input):
        result = self._conv1(input)
        result = self._conv2(result)
        result = self._conv3(result)
        return result


class ProjectionNetworkBlock(keras.layers.Layer):
    def __init__(self, direction, should_reduce_size, should_output_predictions):
        super().__init__()
        self._direction = direction
        self._should_reduce_size = should_reduce_size
        self._should_output_predictions = should_output_predictions

        self._dilated_conv1 = keras.layers.Conv2D(6, 3, padding='same', activation='relu', dilation_rate=2)
        self._dilated_conv2 = keras.layers.Conv2D(6, 3, padding='same', activation='relu', dilation_rate=3)
        self._dilated_conv3 = keras.layers.Conv2D(6, 3, padding='same', activation='relu', dilation_rate=4)
        self._concat1 = keras.layers.Concatenate()
        if should_reduce_size:
            pool_size = (1, 2) if direction == ProjectionDirection.Height else (2, 1)
            self._pooling = keras.layers.MaxPool2D(pool_size, padding='same')
        self._upper_branch_conv = keras.layers.Conv2D(18, 1, activation='relu')
        self._upper_branch_proj = ProjectionLayer(direction, True)
        self._lower_branch_conv = keras.layers.Conv2D(1, 1, activation='sigmoid')
        self._lower_branch_proj = ProjectionLayer(direction, True)
        if should_output_predictions:
            self._prediction_layer = ProjectionLayer(direction, False)
            self._flatten_layer = keras.layers.Flatten()
        self._concat2 = keras.layers.Concatenate()

    def call(self, input):
        middle_result = self._concat1(
            [self._dilated_conv1(input), self._dilated_conv2(input), self._dilated_conv3(input)]
        )
        if self._should_reduce_size:
            middle_result = self._pooling(middle_result)

        upper_result = self._upper_branch_conv(middle_result)
        upper_result = self._upper_branch_proj(upper_result)

        lower_result = self._lower_branch_conv(middle_result)
        if self._should_output_predictions:
            predictions = self._prediction_layer(lower_result)
            predictions = self._flatten_layer(predictions)
        lower_result = self._lower_branch_proj(lower_result)

        result = self._concat2([upper_result, middle_result, lower_result])
        if self._should_output_predictions:
            return [result, predictions]
        return result


class ProjectionNetworkFinalBlock(keras.layers.Layer):
    def __init__(self, direction):
        super().__init__()
        self._dilated_conv1 = keras.layers.Conv2D(6, 3, padding='same', activation='relu', dilation_rate=2)
        self._dilated_conv2 = keras.layers.Conv2D(6, 3, padding='same', activation='relu', dilation_rate=3)
        self._dilated_conv3 = keras.layers.Conv2D(6, 3, padding='same', activation='relu', dilation_rate=4)
        self._concat = keras.layers.Concatenate()
        self._conv1x1 = keras.layers.Conv2D(1, 1, activation='sigmoid')
        self._prediction_layer = ProjectionLayer(direction, False)
        self._flatten_layer = keras.layers.Flatten()

    def call(self, input):
        result = self._concat(
            [self._dilated_conv1(input), self._dilated_conv2(input), self._dilated_conv3(input)]
        )
        result = self._conv1x1(result)
        result = self._prediction_layer(result)
        result = self._flatten_layer(result)
        return result


class ProjectionNetwork(keras.layers.Layer):
    def __init__(self, direction, name):
        super().__init__(name=name)
        self._direction = direction
        self._block1 = ProjectionNetworkBlock(direction, True, False)
        self._block2 = ProjectionNetworkBlock(direction, True, False)
        self._block3 = ProjectionNetworkBlock(direction, True, True)
        self._block4 = ProjectionNetworkBlock(direction, False, True)
        self._block5 = ProjectionNetworkFinalBlock(direction)

    def call(self, input):
        block1_output = self._block1(input)
        block2_output = self._block2(block1_output)
        block3_output, probs1 = self._block3(block2_output)
        block4_output, probs2 = self._block4(block3_output)
        probs3 = self._block5(block4_output)
        return probs1, probs2, probs3


class Model(keras.models.Model):
    def __init__(self, training):
        super().__init__()
        self._normalize_image_layer = keras.layers.experimental.preprocessing.Rescaling(
            scale=1./255)
        self._sfcn = SharedFullyConvolutionalNetwork()
        self._rpn = ProjectionNetwork(ProjectionDirection.Height, 'RPN')
        self._cpn = ProjectionNetwork(ProjectionDirection.Width, 'CPN')
        # Experiment shows, that on ICDAR dataset smoothing postprocessing worsens results.
        self._binarize_horz_splits_layer = BinarizeLayer(0)
        self._binarize_vert_splits_layer = BinarizeLayer(0)

        if training:
            # Adjacency f-measure can't be evaluated efficiently in graph mode.
            self._metric = None
        else:
            self._metric = AdjacencyFMeasure()

    def call(self, input):
        input = self._normalize_image_layer(input)
        sfcn_output = self._sfcn(input)
        horz_split_points_probs1, horz_split_points_probs2, horz_split_points_probs3 = self._rpn(sfcn_output)
        vert_split_points_probs1, vert_split_points_probs2, vert_split_points_probs3 = self._cpn(sfcn_output)
        horz_split_points_binary = self._binarize_horz_splits_layer(horz_split_points_probs3)
        vert_split_points_binary = self._binarize_horz_splits_layer(vert_split_points_probs3)
        return {
            'horz_split_points_probs1': horz_split_points_probs1,
            'horz_split_points_probs2': horz_split_points_probs2,
            'horz_split_points_probs3': horz_split_points_probs3,
            'horz_split_points_binary': horz_split_points_binary,
            'vert_split_points_probs1': vert_split_points_probs1,
            'vert_split_points_probs2': vert_split_points_probs2,
            'vert_split_points_probs3': vert_split_points_probs3,
            'vert_split_points_binary': vert_split_points_binary,
            'markup_table': None
        }

    def compute_metrics(self, input_dict, targets_dict, prediction, sample_weight):
        metric_results = super().compute_metrics(
            input_dict, targets_dict, prediction, sample_weight)

        if self._metric is None:
            return metric_results

        markup_table = Table.from_tensor(tf.squeeze(targets_dict['markup_table'], axis=0))

        h_binary = tf.squeeze(prediction['horz_split_points_binary'], axis=0).numpy()
        v_binary = tf.squeeze(prediction['vert_split_points_binary'], axis=0).numpy()
        
        # Uncomment if you want to know max possible value of metric.
        #h_binary = tf.squeeze(targets_dict['horz_split_points_binary'], axis=0).numpy()
        #v_binary = tf.squeeze(targets_dict['vert_split_points_binary'], axis=0).numpy()

        grid = GridStructureBuilder(markup_table.rect, h_binary, v_binary).build()
        cells = []
        for i in range(grid.get_rows_count()):
            for j in range(grid.get_cols_count()):
                cells.append(Rect(j, i, j+1, i+1))

        self._metric.update_state_eager(markup_table, grid, cells)
        metric_results['adjacency_f_measure'] = self._metric.result()
        
        return metric_results 
