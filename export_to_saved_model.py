import argparse
import shutil
import os

import tensorflow as tf

import split.evaluation
import merge.evaluation


class SplergeModel(tf.Module):
    def __init__(self, split_model, merge_model):
        super().__init__()

        self._split_model = split_model
        self._merge_model = merge_model

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None, 3), dtype=tf.int32)])
    def call(self, image):
        image = tf.expand_dims(image, 0)
        split_outputs = self._split_model(image)
        merge_inputs = {
            'image': image,
            'horz_split_points_probs': split_outputs['horz_split_points_probs3'],
            'vert_split_points_probs': split_outputs['vert_split_points_probs3'],
            'horz_split_points_binary': split_outputs['horz_split_points_binary'],
            'vert_split_points_binary': split_outputs['vert_split_points_binary']
        }  
        merge_outputs = self._merge_model(merge_inputs)
        return {
            'h_positions': merge_outputs['h_positions'],
            'v_positions': merge_outputs['v_positions'],
            'cells_grid_rects': merge_outputs['cells_grid_rects']
        }


def main(args):
    split_model = split.evaluation.load_model(args.split_checkpoint_path, False)
    merge_model = merge.evaluation.load_model(args.merge_checkpoint_path, False)

    model = SplergeModel(split_model, merge_model)
    tf.saved_model.save(model, args.dst_folder_path)
    # Copy custom ops sources and libs to destination folder.
    shutil.copytree(
        './ops', os.path.join(args.dst_folder_path, 'ops'), 
        dirs_exist_ok=True)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Exports trained SPLERGE model to SavedModel format.")
    parser.add_argument('split_checkpoint_path', help='Path to trained SPLIT model checkpoint.')
    parser.add_argument('merge_checkpoint_path', help='Path to trained MERGE model checkpoint.')
    parser.add_argument('dst_folder_path', help='Path to folder where saved model will be stored.')

    main(parser.parse_args())