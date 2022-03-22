import argparse
import shutil
import os

import tensorflow as tf

import merge.model


class ExportableMergeModel(merge.model.Model):
    def __init__(self):
        super().__init__(False)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.int32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        ])
    def call(self, 
            image, horz_split_points_probs, vert_split_points_probs,
            horz_split_points_binary, vert_split_points_binary):

        inputs = {
            'image': tf.expand_dims(image, 0),
            'horz_split_points_probs': tf.expand_dims(horz_split_points_probs, 0),
            'vert_split_points_probs': tf.expand_dims(vert_split_points_probs, 0),
            'horz_split_points_binary': tf.expand_dims(horz_split_points_binary, 0),
            'vert_split_points_binary': tf.expand_dims(vert_split_points_binary, 0)
        }    
        outputs = super().call(inputs)
        return {
            'merge_right_probs': tf.squeeze(outputs['merge_right_probs2'], 0),
            'merge_down_probs': tf.squeeze(outputs['merge_down_probs2'], 0)
        }


def main(args):
    model = ExportableMergeModel()
    model(
        tf.zeros(shape=(1, 1, 3), dtype=tf.int32),
        tf.zeros(shape=(1,)), tf.zeros(shape=(1,)),
        tf.zeros(shape=(1,), dtype=tf.int32), tf.zeros(shape=(1,), dtype=tf.int32)
    )
    model.load_weights(args.checkpoint_path)

    tf.saved_model.save(model, args.dst_folder_path)
    # Copy custom ops sources and libs to destination folder.
    shutil.copytree(
        './merge/ops', os.path.join(args.dst_folder_path, 'ops'), 
        dirs_exist_ok=True)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Exports trained model checkpoint to SavedModel format.")
    parser.add_argument('checkpoint_path', help='Path to trained model checkpoint.')
    parser.add_argument('dst_folder_path', help='Path to folder where saved model will be stored.')

    main(parser.parse_args())