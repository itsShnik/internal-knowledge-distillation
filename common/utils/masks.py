#----------------------------------------
#--------- Library imports --------------
#----------------------------------------
import random

def generate_additional_head_masks_to_res50():
    """
    There are 23 blocks in ResNet101, 6 in ResNet50,
    We need to return a list containing 6 indices,
    conditions: It must contain 0, 22 and rest can be random
    """
    first_layer = [*range(3)]
    second_layer = [*range(4)]
    fourth_layer = [*range(3)]

    # Third layer
    first_index = [0]
    last_index = [22]
    rest_of_the_indices = random.sample(range(1,22), 4)
    third_layer = first_index + rest_of_the_indices + last_index

    masks = [first_layer, second_layer, third_layer, fourth_layer]

    return masks

def generate_additional_head_masks_to_res24():
    """
    From every group (layer) take the first and the last block
    """
    first_layer = [0,2]
    second_layer = [0,3]
    third_layer = [0,22]
    fourth_layer = [0,2]

    masks = [first_layer, second_layer, third_layer, fourth_layer]

    return masks

def generate_additional_head_masks_to_res18():
    """
    From every group (layer) take the first and the last block
    """
    first_layer = [0,2]
    second_layer = [0,3]
    third_layer = [0,5]
    fourth_layer = [0,2]

    masks = [first_layer, second_layer, third_layer, fourth_layer]

    return masks


