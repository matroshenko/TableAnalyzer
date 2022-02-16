import tensorflow.keras as keras


def get_losses_dict():
    return {
        'horz_split_points_probs1': keras.losses.BinaryCrossentropy(from_logits=False),
        'horz_split_points_probs2': keras.losses.BinaryCrossentropy(from_logits=False),
        'horz_split_points_probs3': keras.losses.BinaryCrossentropy(from_logits=False),
        'vert_split_points_probs1': keras.losses.BinaryCrossentropy(from_logits=False),
        'vert_split_points_probs2': keras.losses.BinaryCrossentropy(from_logits=False),
        'vert_split_points_probs3': keras.losses.BinaryCrossentropy(from_logits=False),
        'markup_table': None
    }

def get_losses_weights():
    return {
        'horz_split_points_probs1': 0.1,
        'horz_split_points_probs2': 0.25,
        'horz_split_points_probs3': 1,
        'vert_split_points_probs1': 0.1,
        'vert_split_points_probs2': 0.25,
        'vert_split_points_probs3': 1,
        'markup_table': 0
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
            'vert_split_points_binary': vert_split_points_mask,
            'markup_table': element['markup_table']   
        }
    )