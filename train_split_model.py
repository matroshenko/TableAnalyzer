import argparse
from enum import Enum
from datetime import datetime

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds

from datasets.ICDAR.ICDAR import Icdar
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
        image,
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

def build_data_pipeline(ds, target, max_samples_count):
    if max_samples_count is not None:
        assert max_samples_count > 0
        ds = ds.take(max_samples_count)
    
    ds = ds.map(convert_ds_element_to_tuple)
    if target == Target.Train:
        ds = ds.shuffle(128)
    ds = ds.batch(1)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def get_tensorboard_callback():
    now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    root_logdir = '/tmp/tf_logs'
    logdir = '{}/run-{}/'.format(root_logdir, now)
    return tf.keras.callbacks.TensorBoard(log_dir=logdir)

def main(args):
    # For reproducible results.
    tf.random.set_seed(42)
    # For debugging it's better to see full stack trace.
    tf.debugging.disable_traceback_filtering()

    model = Model()
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        args.initial_learning_rate,
        decay_steps=80000,
        decay_rate=0.075,
        staircase=True)

    # Unfortunately you need to run model in eager mode,
    # because some custom layers (BinarizeLayer and IntervalwiseFMeasure)
    # can't run in graph mode.
    model.compile(
        keras.optimizers.Adam(lr_schedule), 
        loss=get_losses_dict(), 
        loss_weights=get_losses_weights(),
        metrics=get_metrics_dict(), run_eagerly=True)

    ds_train, ds_test = tfds.load(
        'ICDAR',
        split=['train[:90%]', 'train[90%:]']
    )
    ds_train = build_data_pipeline(ds_train, Target.Train, args.max_samples_count)
    ds_test = build_data_pipeline(ds_test, Target.Test, args.max_samples_count)

    model.fit(
        ds_train, epochs=args.epochs_count, validation_data=ds_test,
        callbacks=[get_tensorboard_callback()])
    model.save_weights(args.result_file_path, save_format='h5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trains SPLIT model.")
    parser.add_argument('result_file_path', help='Path to the file, where trained model will be serialized.')
    parser.add_argument('--epochs_count', default=10, type=int, help='Number of epochs to train.')
    parser.add_argument('--initial_learning_rate', default=0.00075, help='Initial value of learning rate.')
    parser.add_argument('--max_samples_count', default=None, type=int, 
        help='Max count of samples to train/test. May be used for debug purposes.')
    main(parser.parse_args())