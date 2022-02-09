import argparse
import os.path

import tensorflow as tf
import tensorflow_datasets as tfds

from split.model import Model
from datasets.ICDAR.ICDAR import IcdarSplit


def run_model_on_random_input(model):
    random_image = tf.random.uniform(shape=(1, 32, 32, 3), minval=0, maxval=256, dtype='int32')
    model(random_image)

def load_model(model_file_path):
    assert os.path.exists(model_file_path)
    model = Model()
    run_model_on_random_input(model)
    model.load_weights(model_file_path)
    model.compile(run_eagerly=True)
    return model

def convert_ds_element_to_tuple(element):    
    return (
        element['image'],
        {
            'horz_split_points_binary': element['horz_split_points_mask'],   
            'vert_split_points_binary': element['vert_split_points_mask'],
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

    ds = tfds.load('icdar_split', split=args.split)
    ds = build_data_pipeline(ds)

    model.evaluate(ds)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluates trained SPLIT model on ICDAR dataset.")
    parser.add_argument('model_file_path', help='Path to trained model file.')
    parser.add_argument('split', choices=['train', 'test'], help='Name of the dataset split to evaluate.')

    main(parser.parse_args())