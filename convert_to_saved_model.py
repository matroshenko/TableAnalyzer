import argparse

import split.evaluation
import merge.evaluation


def main(args):
    module = split.evaluation if args.model_type == 'SPLIT' else merge.evaluation
    model = module.load_model(args.model_file_path, False)
    model.save(args.dst_folder_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts trained model to SavedModel format.")
    parser.add_argument('model_type', help='Type of model', choices=['SPLIT', 'MERGE'])
    parser.add_argument('model_file_path', help='Path to trained model checkpoint.')
    parser.add_argument('dst_folder_path', help='Path to folder where saved model will be stored.')

    main(parser.parse_args())