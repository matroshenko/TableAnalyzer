import tensorflow as tf
import tensorflow.keras as keras

from projection_layer import ProjectionLayer, ProjectionDirection

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
    def __init__(self, direction, should_reduce_size, should_output_predictions):
        super().__init__()
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
    def __init__(self, direction):
        super().__init__()
        self._block1 = ProjectionNetworkBlock(direction, True, False)
        self._block2 = ProjectionNetworkBlock(direction, True, False)
        self._block3 = ProjectionNetworkBlock(direction, True, True)
        self._block4 = ProjectionNetworkBlock(direction, False, True)
        self._block5 = ProjectionNetworkFinalBlock(direction)

    def call(self, x):
        x = self._block1(x)
        x = self._block2(x)
        x, predictions1 = self._block3(x)
        x, predictions2 = self._block4(x)
        predictions3 = self._block5(x)
        return predictions1, predictions2, predictions3


class Model(keras.models.Model):
    def __init__(self):
        super().__init__()
        self._sfcn = SharedFullyConvolutionalNetwork()
        self._rpn = ProjectionNetwork(ProjectionDirection.Height)
        self._cpn = ProjectionNetwork(ProjectionDirection.Width)

    def call(self, input):
        sfcn_output = self._sfcn(input)
        horz_split_points_probs1, horz_split_points_probs2, horz_split_points_probs3 = self._rpn(sfcn_output)
        vert_split_points_probs1, vert_split_points_probs2, vert_split_points_probs3 = self._cpn(sfcn_output)
        return (
            horz_split_points_probs1, horz_split_points_probs2, horz_split_points_probs3,
            vert_split_points_probs1, vert_split_points_probs2, vert_split_points_probs3
        )
