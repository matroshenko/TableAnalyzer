import argparse
from enum import Enum
from datetime import datetime

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds

from datasets.ICDAR.ICDAR import IcdarMerge
from datasets.FinTabNet.FinTabNet import FinTabNetMerge
from merge.model import Model

def get_losses_dict():
    return {
        'merge_down_probs1': keras.losses.BinaryCrossentropy(from_logits=False, axis=(1, 2)),
        'merge_down_probs2': keras.losses.BinaryCrossentropy(from_logits=False, axis=(1, 2)),
        'merge_right_probs1': keras.losses.BinaryCrossentropy(from_logits=False, axis=(1, 2)),
        'merge_right_probs2': keras.losses.BinaryCrossentropy(from_logits=False, axis=(1, 2)),
        'markup_table': None
    }

def get_losses_weights():
    return {
        'merge_down_probs1': 0.5,
        'merge_down_probs2': 1,
        'merge_right_probs1': 0.5,
        'merge_right_probs2': 1,
        'markup_table': 0
    }

def get_metrics_dict():
    return {}

def convert_ds_element_to_tuple(element):
    merge_down_mask = element['merge_down_mask']
    merge_right_mask = element['merge_right_mask']

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
            'merge_down_probs1': merge_down_mask,
            'merge_down_probs2': merge_down_mask,
            'merge_right_probs1': merge_right_mask,
            'merge_right_probs2': merge_right_mask,
            'markup_table': element['markup_table']  
        }
    )

def build_data_pipeline(ds, max_samples_count):
    if max_samples_count is not None:
        assert max_samples_count > 0
        ds = ds.take(max_samples_count)
    
    ds = ds.map(convert_ds_element_to_tuple)
    ds = ds.shuffle(128)
    ds = ds.batch(1)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def get_tensorboard_callback():
    now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    root_logdir = '/tmp/tf_logs/merge'
    logdir = '{}/run-{}/'.format(root_logdir, now)
    return tf.keras.callbacks.TensorBoard(log_dir=logdir)

def main(args):
    # For reproducible results.
    tf.random.set_seed(42)
    # For debugging it's better to see full stack trace.
    # tf.debugging.disable_traceback_filtering()

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

    ds_train = tfds.load(args.dataset_name, split='train')
    ds_train = build_data_pipeline(ds_train, args.max_samples_count)

    model.fit(
        ds_train, epochs=args.epochs_count,
        callbacks=[get_tensorboard_callback()])
    model.save_weights(args.result_file_path, save_format='h5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trains MERGE model.")
    parser.add_argument('dataset_name', help='Name of the dataset to train on.', 
        choices=['icdar_merge', 'fin_tab_net_merge'])
    parser.add_argument('result_file_path', help='Path to the file, where trained model will be serialized.')
    parser.add_argument('--epochs_count', default=10, type=int, help='Number of epochs to train.')
    parser.add_argument('--initial_learning_rate', default=0.00075, help='Initial value of learning rate.')
    parser.add_argument('--max_samples_count', default=None, type=int, 
        help='Max count of samples to train/test. May be used for debug purposes.')
    main(parser.parse_args())