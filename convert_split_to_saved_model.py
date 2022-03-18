import argparse
import os

import tensorflow as tf

import split.model


class ExportableSplitModel(split.model.Model):
    def __init__(self):
        super().__init__(False)

    @tf.function(input_signature=[tf.TensorSpec(shape=(1, None, None, 3), dtype=tf.int32)])
    def call(self, input):
        return super().call(input)


def main(args):
    # We can't export original model, 
    # because it won't run on images with dynamic size.
    model = ExportableSplitModel()
    model(tf.zeros(shape=(1, 1, 1, 3), dtype=tf.int32))
    model.load_weights(args.model_file_path)

    tf.saved_model.save(model, args.dst_folder_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts trained model to SavedModel format.")
    parser.add_argument('model_file_path', help='Path to trained model checkpoint.')
    parser.add_argument('dst_folder_path', help='Path to folder where saved model will be stored.')

    main(parser.parse_args())