import argparse
import os.path

import tensorflow as tf
import tensorflow_datasets as tfds

from merge.model import Model
from datasets.ICDAR.ICDAR import IcdarMerge
from datasets.FinTabNet.FinTabNet import FinTabNetMerge


def run_model_on_random_input(model):
    batch_size = 1
    height = 100
    width = 200
    inputs = {
        'image': tf.random.uniform(shape=(batch_size, height, width, 3), minval=0, maxval=256, dtype='int32'),
        'horz_split_points_probs': tf.random.uniform(shape=(batch_size, height), dtype='float32'),
        'vert_split_points_probs': tf.random.uniform(shape=(batch_size, width), dtype='float32'),
        'horz_split_points_binary': tf.random.uniform(shape=(batch_size, height), minval=0, maxval=2, dtype='int32'),
        'vert_split_points_binary': tf.random.uniform(shape=(batch_size, width), minval=0, maxval=2, dtype='int32')
    }
    model(inputs)

def load_model(model_file_path):
    assert os.path.exists(model_file_path)
    model = Model()
    run_model_on_random_input(model)
    model.load_weights(model_file_path)
    model.compile(run_eagerly=True)
    return model

def convert_ds_element_to_tuple(element):
    input_keys = [
        'image', 
        'horz_split_points_probs', 
        'vert_split_points_probs',
        'horz_split_points_binary',
        'vert_split_points_binary'
    ]
    return (
        {key: element[key] for key in input_keys},
        {
            'markup_table': element['markup_table']  
        }
    )

def build_data_pipeline(ds):
    ds = ds.map(convert_ds_element_to_tuple)
    ds = ds.batch(1)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def main(args):
    model = load_model(args.model_file_path)

    ds = tfds.load(args.dataset_name + '_merge', split=args.split)
    ds = build_data_pipeline(ds)

    model.evaluate(ds)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluates trained MERGE model on ICDAR dataset.")
    parser.add_argument('model_file_path', help='Path to trained model file.')
    parser.add_argument('dataset_name', help='Name of the dataset to evaluate on.', 
        choices=['icdar', 'fin_tab_net'])
    parser.add_argument('split', choices=['train', 'test'], help='Name of the dataset split to evaluate.')

    main(parser.parse_args())