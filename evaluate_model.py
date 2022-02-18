import argparse
import os.path

import tensorflow as tf
import tensorflow_datasets as tfds

import split.evaluation
import merge.evaluation
from datasets.ICDAR.ICDAR import IcdarSplit
from datasets.FinTabNet.FinTabNet import FinTabNetSplit


def main(args):
    module = split.evaluation if args.model_type == 'SPLIT' else merge.evaluation
    model = module.load_model(args.model_file_path, True)

    ds_suffix = '_split' if args.model_type == 'SPLIT' else '_merge'
    ds = tfds.load(args.dataset_name + ds_suffix, split=args.split)
    ds = ds.map(module.convert_ds_element_to_tuple)
    ds = ds.batch(1)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    model.evaluate(ds)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluates trained model of specific type on specific dataset.")
    parser.add_argument('model_type', help='Type of model', choices=['SPLIT', 'MERGE'])
    parser.add_argument('model_file_path', help='Path to trained model file.')
    parser.add_argument('dataset_name', help='Name of the dataset to evaluate on.', 
        choices=['icdar', 'fin_tab_net'])
    parser.add_argument('split', choices=['train', 'test'], help='Name of the dataset split to evaluate.')

    main(parser.parse_args())