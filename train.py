import argparse
from enum import Enum

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds

import datasets.ICDAR
from split.model import Model
from split.intervalwise_f_measure import IntervalwiseFMeasure

def get_losses_dict():
    return {
        'horz_split_points_probs1': keras.losses.BinaryCrossentropy(from_logits=False),
        'horz_split_points_probs2': keras.losses.BinaryCrossentropy(from_logits=False),
        'horz_split_points_probs3': keras.losses.BinaryCrossentropy(from_logits=False),
        'vert_split_points_probs1': keras.losses.BinaryCrossentropy(from_logits=False),
        'vert_split_points_probs2': keras.losses.BinaryCrossentropy(from_logits=False),
        'vert_split_points_probs3': keras.losses.BinaryCrossentropy(from_logits=False),
    }

def get_losses_weights():
    return {
        'horz_split_points_probs1': 0.1,
        'horz_split_points_probs2': 0.25,
        'horz_split_points_probs3': 1,
        'vert_split_points_probs1': 0.1,
        'vert_split_points_probs2': 0.25,
        'vert_split_points_probs3': 1,
    }

def get_metrics_dict():
    return {
        'horz_split_points_binary': IntervalwiseFMeasure(),
        'vert_split_points_binary': IntervalwiseFMeasure(),
    }

def convert_ds_element_to_tuple(element):
    image = element['image']
    horz_split_points_mask = element['horz_split_points_mask']
    vert_split_points_mask = element['vert_split_points_mask']
    return (
        {
            'image': image,
            'image_height': tf.size(horz_split_points_mask),
            'image_width': tf.size(vert_split_points_mask)
        },
        {
            'horz_split_points_probs1': horz_split_points_mask,
            'horz_split_points_probs2': horz_split_points_mask,
            'horz_split_points_probs3': horz_split_points_mask,
            'horz_split_points_binary': horz_split_points_mask,
            'vert_split_points_probs1': vert_split_points_mask,
            'vert_split_points_probs2': vert_split_points_mask,
            'vert_split_points_probs3': vert_split_points_mask,    
            'vert_split_points_binary': vert_split_points_mask   
        }
    )

class Target(Enum):
    Train = 0,
    Test = 1

def build_data_pipeline(ds, target):
    ds = ds.map(convert_ds_element_to_tuple)
    if target == Target.Train:
        ds = ds.shuffle(128, seed=42)
    ds = ds.batch(1)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def main(args):
    model = Model()
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        args.initial_learning_rate,
        decay_steps=80000,
        decay_rate=0.075,
        staircase=True)
    model.compile(
        keras.optimizers.Adam(lr_schedule), 
        loss=get_losses_dict(), 
        loss_weights=get_losses_weights(),
        metrics=get_metrics_dict(), run_eagerly=True)

    ds_train, ds_test = tfds.load(
        'ICDAR',
        split=['train[:90%]', 'train[90%:]']
    )
    ds_train = build_data_pipeline(ds_train, Target.Train)
    ds_test = build_data_pipeline(ds_test, Target.Test)

    model.fit(ds_train, epochs=1, validation_data=ds_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trains SPLIT model.")
    parser.add_argument('result_file_path', help='Path to the file, where trained model will be serialized.')
    parser.add_argument('--initial_learning_rate', default=0.00075, help='Initial value of learning rate.')
    main(parser.parse_args())