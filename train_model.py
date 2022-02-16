import argparse
from enum import Enum
from datetime import datetime
import os

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds

from datasets.ICDAR.ICDAR import IcdarMerge
from datasets.FinTabNet.FinTabNet import FinTabNetMerge
import merge
import merge.training
import split
import split.training


def build_data_pipeline(ds, max_samples_count):
    if max_samples_count is not None:
        assert max_samples_count > 0
        ds = ds.take(max_samples_count)
    
    ds = ds.map(merge.training.convert_ds_element_to_tuple)
    ds = ds.shuffle(128)
    ds = ds.batch(1)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def get_tensorboard_callback(model_type):
    now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    root_logdir = os.path.join('/tmp/tf_logs', model_type)
    logdir = '{}/run-{}/'.format(root_logdir, now)
    return tf.keras.callbacks.TensorBoard(log_dir=logdir)

def main(args):
    # For reproducible results.
    tf.random.set_seed(42)
    # For debugging it's better to see full stack trace.
    # tf.debugging.disable_traceback_filtering()

    module = split if args.model_type == 'SPLIT' else merge

    model = module.model.Model()
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        args.initial_learning_rate,
        decay_steps=80000,
        decay_rate=0.075,
        staircase=True)

    model.compile(
        keras.optimizers.Adam(lr_schedule), 
        loss=module.training.get_losses_dict(), 
        loss_weights=module.training.get_losses_weights(),
        run_eagerly=True)

    ds_suffix = '_split' if args.model_type == 'SPLIT' else '_merge'
    ds_train = tfds.load(args.dataset_name + ds_suffix, split='train')
    if args.max_samples_count is not None:
        assert args.max_samples_count > 0
        ds = ds.take(args.max_samples_count)
    
    ds = ds.map(module.training.convert_ds_element_to_tuple)
    ds = ds.shuffle(128)
    ds = ds.batch(1)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    model.fit(
        ds_train, epochs=args.epochs_count,
        callbacks=[get_tensorboard_callback(args.model_type)])
    model.save_weights(args.result_file_path, save_format='h5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trains model of specific type.")
    parser.add_argument('model_type', help='Type of model', choices=['SPLIT', 'MERGE'])
    parser.add_argument('dataset_name', help='Name of the dataset to train on.', 
        choices=['icdar', 'fin_tab_net'])
    parser.add_argument('result_file_path', help='Path to the file, where trained model will be serialized.')
    parser.add_argument('--epochs_count', default=10, type=int, help='Number of epochs to train.')
    parser.add_argument('--initial_learning_rate', default=0.00075, help='Initial value of learning rate.')
    parser.add_argument('--max_samples_count', default=None, type=int, 
        help='Max count of samples to train/test. May be used for debug purposes.')
    main(parser.parse_args())