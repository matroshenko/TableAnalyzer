import tensorflow.keras as keras
import tensorflow as tf


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

def has_more_than_one_row_and_column(element):
    h_mask = tf.cast(element['horz_split_points_binary'], tf.bool)
    v_mask = tf.cast(element['vert_split_points_binary'], tf.bool)
    return tf.reduce_any(h_mask) and tf.reduce_any(v_mask)

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