import argparse
import shutil
import os

import tensorflow as tf

import split.model


class ExportableSplitModel(split.model.Model):
    def __init__(self):
        super().__init__(False)

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None, 3), dtype=tf.int32)])
    def call(self, image):
        outputs = super().call(tf.expand_dims(image, 0))
        return {
            'horz_split_points_probs': tf.squeeze(outputs['horz_split_points_probs3'], 0),
            'horz_split_points_binary': tf.squeeze(outputs['horz_split_points_binary'], 0),
            'vert_split_points_probs': tf.squeeze(outputs['vert_split_points_probs3'], 0),
            'vert_split_points_binary': tf.squeeze(outputs['vert_split_points_binary'], 0)
        }


def main(args):
    model = ExportableSplitModel()
    model(tf.zeros(shape=(1, 1, 3), dtype=tf.int32))
    model.load_weights(args.checkpoint_path)

    tf.saved_model.save(model, args.dst_folder_path)
    # Copy custom ops sources and libs to destination folder.
    shutil.copytree(
        './split/ops', os.path.join(args.dst_folder_path, 'ops'), 
        dirs_exist_ok=True)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Exports trained model checkpoint to SavedModel format.")
    parser.add_argument('checkpoint_path', help='Path to trained model checkpoint.')
    parser.add_argument('dst_folder_path', help='Path to folder where saved model will be stored.')

    main(parser.parse_args())