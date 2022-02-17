import os
import tensorflow as tf

from merge.model import Model


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
    model = Model(False)
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