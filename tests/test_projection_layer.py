from unittest import TestCase, main

import tensorflow as tf

import context
from split.projection_layer import ProjectionDirection, ProjectionLayer


class ProjectionLayerTestCase(TestCase):
    def setUp(self):
        self.input = tf.constant([[
            [[1, 2], [3, 4], [5, 6]],
            [[7, 8], [9, 10], [11, 12]],
        ]], dtype = 'float32')

    def test_project_on_height_no_broadcast(self):
        expected_output = tf.constant([[
            [[3, 4]],
            [[9, 10]]
        ]], dtype = 'float32')
        layer = ProjectionLayer(ProjectionDirection.Height, False)
        output = layer(self.input)
        self.assertTrue(tf.reduce_all(expected_output == output))

    def test_project_on_height_broadcast(self):
        expected_output = tf.constant([[
            [[3, 4], [3, 4], [3, 4]],
            [[9, 10], [9, 10], [9, 10]]
        ]], dtype = 'float32')
        layer = ProjectionLayer(ProjectionDirection.Height, True)
        output = layer(self.input)
        self.assertTrue(tf.reduce_all(expected_output == output))

    def test_project_on_width_no_broadcast(self):
        expected_output = tf.constant([[
            [[4, 5], [6, 7], [8, 9]]
        ]], dtype = 'float32')
        layer = ProjectionLayer(ProjectionDirection.Width, False)
        output = layer(self.input)
        self.assertTrue(tf.reduce_all(expected_output == output))

    def test_project_on_width_broadcast(self):
        expected_output = tf.constant([[
            [[4, 5], [6, 7], [8, 9]],
            [[4, 5], [6, 7], [8, 9]]
        ]], dtype = 'float32')
        layer = ProjectionLayer(ProjectionDirection.Width, True)
        output = layer(self.input)
        self.assertTrue(tf.reduce_all(expected_output == output))

if __name__ == '__main__':
    main()