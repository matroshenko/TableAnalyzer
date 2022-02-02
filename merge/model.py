import tensorflow as tf
import tensorflow.keras as keras

from merge.grid_pooling_layer import GridPoolingLayer

class SharedFullyConvolutionalNetwork(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        # Original paper suggests to use kernel size = 7,
        # which leads to excessive memory consumption.
        self._conv1 = keras.layers.Conv2D(18, 3, padding='same', activation='relu')
        self._pool1 = keras.layers.MaxPool2D()
        self._conv2 = keras.layers.Conv2D(18, 3, padding='same', activation='relu')
        self._conv3 = keras.layers.Conv2D(18, 3, padding='same', activation='relu')
        self._conv4 = keras.layers.Conv2D(18, 3, padding='same', activation='relu')
        self._pool2 = keras.layers.MaxPool2D()

    def call(self, input):
        result = self._conv1(input)
        result = self._pool1(result)
        result = self._conv2(result)
        result = self._conv3(result)
        result = self._conv4(result)
        result = self._pool2(result)
        return result


class GridPoolingNetworkBlock(keras.layers.Layer):
    def __init__(self, should_output_predictions):
        super().__init__()
        self._should_output_predictions = should_output_predictions

        self._dilated_conv1 = keras.layers.Conv2D(6, 3, padding='same', activation='relu', dilation_rate=1)
        self._dilated_conv2 = keras.layers.Conv2D(6, 3, padding='same', activation='relu', dilation_rate=2)
        self._dilated_conv3 = keras.layers.Conv2D(6, 3, padding='same', activation='relu', dilation_rate=3)
        self._concat1 = keras.layers.Concatenate()

        self._upper_branch_conv = keras.layers.Conv2D(18, 1, activation='relu')
        self._upper_branch_pool = GridPoolingLayer(True)
        self._lower_branch_conv = keras.layers.Conv2D(1, 1, activation='sigmoid')
        self._lower_branch_pool = GridPoolingLayer(True)
        if should_output_predictions:
            self._prediction_layer = GridPoolingLayer(False)
            self._flatten_layer = keras.layers.Flatten()

        self._concat2 = keras.layers.Concatenate()

    def call(self, input, h_positions, v_positions):
        middle_result = self._concat1(
            [self._dilated_conv1(input), self._dilated_conv2(input), self._dilated_conv3(input)]
        )

        upper_result = self._upper_branch_conv(middle_result)
        upper_result = self._upper_branch_pool(upper_result, h_positions, v_positions)

        lower_result = self._lower_branch_conv(middle_result)
        if self._should_output_predictions:
            predictions = self._prediction_layer(lower_result, h_positions, v_positions)
            predictions = self._flatten_layer(predictions)
        lower_result = self._lower_branch_pool(lower_result, h_positions, v_positions)

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
        self._flatten_layer = keras.layers.Flatten()

    def call(self, input, h_positions, v_positions):
        result = self._concat(
            [self._dilated_conv1(input), self._dilated_conv2(input), self._dilated_conv3(input)]
        )
        result = self._conv1x1(result)
        result = self._prediction_layer(result, h_positions, v_positions)
        result = self._flatten_layer(result)
        return result
