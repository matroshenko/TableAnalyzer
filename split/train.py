import argparse

import tensorflow.keras as keras
import tensorflow_datasets as tfds

import datasets.ICDAR
from model import Model

def main(args):
    model = Model()
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        args.initial_learning_rate,
        decay_steps=80000,
        decay_rate=0.075,
        staircase=True)
    model.compile(
        keras.optimizers.Adam(lr_schedule), 
        loss=[keras.losses.BinaryCrossentropy(from_logits=False) for _ in range(6)], 
        loss_weights=[0.1, 0.25, 1, 0.1, 0.25, 1])
    (ds_train, ds_test), ds_info = tfds.load(
        'ICDAR',
        split=['train[:90%]', 'train[90%:]'],
        with_info=True
    )
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trains SPLIT model.")
    parser.add_argument('result_file_path', help='Path to the file, where trained model will be serialized.')
    parser.add_argument('--initial_learning_rate', default=0.00075, help='Initial value of learning rate.')
    main(parser.parse_args())