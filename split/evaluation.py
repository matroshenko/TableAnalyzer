import os
import tensorflow as tf

from split.model import Model


def run_model_on_random_input(model):
    random_image = tf.random.uniform(shape=(1, 32, 32, 3), minval=0, maxval=256, dtype='int32')
    model(random_image)

def load_model(model_file_path):
    assert os.path.exists(model_file_path)
    model = Model(False)
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