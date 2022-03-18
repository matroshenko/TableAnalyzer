import argparse
import os

import tensorflow as tf

import split.model
import split.evaluation


def main(args):
    model = split.evaluation.load_model(args.model_file_path, False)
    tf.saved_model.save(model, args.dst_folder_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts trained model to SavedModel format.")
    parser.add_argument('model_file_path', help='Path to trained model checkpoint.')
    parser.add_argument('dst_folder_path', help='Path to folder where saved model will be stored.')

    main(parser.parse_args())