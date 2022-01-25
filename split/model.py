import tensorflow as tf
import tensorflow.keras as keras

from projection_layer import ProjectionLayer, ProjectionDirection
from binarize_layer import BinarizeLayer


def reduce_shape_by_half(direction, input_height, input_width):
    if direction == ProjectionDirection.Height:
        return input_height, tf.cast(tf.math.floor(input_width / 2), 'int32')
    return tf.cast(tf.math.floor(input_height / 2), 'int32'), input_width


class SharedFullyConvolutionalNetwork(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        # Original paper suggests to use kernel size = 7,
        # which leads to excessive memory consumption.
        self._conv1 = keras.layers.Conv2D(48, 3, padding='same', activation='relu')
        self._conv2 = keras.layers.Conv2D(48, 3, padding='same', activation='relu')
        self._conv3 = keras.layers.Conv2D(48, 3, padding='same', activation='relu', dilation_rate=2)

    def call(self, input):
        result = self._conv1(input)
        result = self._conv2(result)
        result = self._conv3(result)
        return result


class ProjectionNetworkBlock(keras.layers.Layer):
    def __init__(self, direction, should_reduce_size, should_output_predictions, ouput_name):
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
            self._pooling = keras.layers.MaxPool2D(pool_size)
        self._upper_branch_conv = keras.layers.Conv2D(18, 1, activation='relu')
        self._upper_branch_proj = ProjectionLayer(direction, True)
        self._lower_branch_conv = keras.layers.Conv2D(1, 1, activation='sigmoid')
        self._lower_branch_proj = ProjectionLayer(direction, True)
        if should_output_predictions:
            self._prediction_layer = ProjectionLayer(direction, False)
            self._flatten_layer = keras.layers.Flatten(name=ouput_name)
        self._concat2 = keras.layers.Concatenate()

    def call(self, input, input_height, input_width):
        middle_result = self._concat1(
            [self._dilated_conv1(input), self._dilated_conv2(input), self._dilated_conv3(input)]
        )
        if self._should_reduce_size:
            middle_result = self._pooling(middle_result)
            input_height, input_width = reduce_shape_by_half(
                self._direction, input_height, input_width)

        upper_result = self._upper_branch_conv(middle_result)
        upper_result = self._upper_branch_proj(upper_result, input_height, input_width)

        lower_result = self._lower_branch_conv(middle_result)
        if self._should_output_predictions:
            predictions = self._prediction_layer(lower_result, input_height, input_width)
            predictions = self._flatten_layer(predictions)
        lower_result = self._lower_branch_proj(lower_result, input_height, input_width)

        result = self._concat2([upper_result, middle_result, lower_result])
        if self._should_output_predictions:
            return [result, predictions]
        return result


class ProjectionNetworkFinalBlock(keras.layers.Layer):
    def __init__(self, direction, output_name):
        super().__init__()
        self._dilated_conv1 = keras.layers.Conv2D(6, 3, padding='same', activation='relu', dilation_rate=2)
        self._dilated_conv2 = keras.layers.Conv2D(6, 3, padding='same', activation='relu', dilation_rate=3)
        self._dilated_conv3 = keras.layers.Conv2D(6, 3, padding='same', activation='relu', dilation_rate=4)
        self._concat = keras.layers.Concatenate()
        self._conv1x1 = keras.layers.Conv2D(1, 1, activation='sigmoid')
        self._prediction_layer = ProjectionLayer(direction, False)
        self._flatten_layer = keras.layers.Flatten(name=output_name)

    def call(self, input, input_height, input_width):
        result = self._concat(
            [self._dilated_conv1(input), self._dilated_conv2(input), self._dilated_conv3(input)]
        )
        result = self._conv1x1(result)
        result = self._prediction_layer(result, input_height, input_width)
        result = self._flatten_layer(result)
        return result


class ProjectionNetwork(keras.layers.Layer):
    def __init__(self, direction, output_name1, output_name2, output_name3):
        super().__init__()
        self._direction = direction
        self._block1 = ProjectionNetworkBlock(direction, True, False, None)
        self._block2 = ProjectionNetworkBlock(direction, True, False, None)
        self._block3 = ProjectionNetworkBlock(direction, True, True, output_name1)
        self._block4 = ProjectionNetworkBlock(direction, False, True, output_name2)
        self._block5 = ProjectionNetworkFinalBlock(direction, output_name3)

    def call(self, input, input_height, input_width):
        block1_output = self._block1(input, input_height, input_width)
        block1_output_height, block1_output_width = reduce_shape_by_half(
            self._direction, input_height, input_width)

        block2_output = self._block2(block1_output, block1_output_height, block1_output_width)
        block2_output_height, block2_output_width = reduce_shape_by_half(
            self._direction, block1_output_height, block1_output_width)

        block3_output, probs1 = self._block3(block2_output, block2_output_height, block2_output_width)
        block3_output_height, block3_output_width = reduce_shape_by_half(
            self._direction, block2_output_height, block2_output_width)

        block4_output, probs2 = self._block4(block3_output, block3_output_height, block3_output_width)
        # block4 does not reduce shape
        block4_output_height, block4_output_width = block3_output_height, block3_output_width

        probs3 = self._block5(block4_output, block4_output_height, block4_output_width)
        return probs1, probs2, probs3

class Model(keras.models.Model):
    def __init__(self):
        super().__init__()
        self._normalize_image_layer = keras.layers.experimental.preprocessing.Rescaling(
            scale=1./255)
        self._sfcn = SharedFullyConvolutionalNetwork()
        self._rpn = ProjectionNetwork(
            ProjectionDirection.Height, 
            'horz_split_points_probs1', 'horz_split_points_probs2', 'horz_split_points_probs3')
        self._cpn = ProjectionNetwork(
            ProjectionDirection.Width, 
            'vert_split_points_probs1', 'vert_split_points_probs2', 'vert_split_points_probs3')
        self._binarize_horz_splits_layer = BinarizeLayer(0.75, 'horz_split_points_binary')
        self._binarize_vert_splits_layer = BinarizeLayer(0.75, 'vert_split_points_binary')

    def call(self, inputs):
        input = inputs['image']
        input_height = inputs['image_height']
        input_width = inputs['image_width']
        input = self._normalize_image_layer(input)
        sfcn_output = self._sfcn(input)
        horz_split_points_probs1, horz_split_points_probs2, horz_split_points_probs3 = self._rpn(
            sfcn_output, input_height, input_width)
        vert_split_points_probs1, vert_split_points_probs2, vert_split_points_probs3 = self._cpn(
            sfcn_output, input_height, input_width)
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
            'vert_split_points_binary': vert_split_points_binary
        }
