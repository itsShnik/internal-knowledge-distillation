#----------------------------------------
#--------- Torch related imports --------
#----------------------------------------
import torch

#----------------------------------------
#--------- Library Imports --------------
#----------------------------------------
import random

def to_cuda(batch):
    # convert batch: tuple to batch: list
    batch = list(batch)

    for i in range(len(batch)):
        assert isinstance(batch[i], torch.Tensor), "Each element of batch is not a tensor"
        batch[i] = batch[i].cuda(non_blocking=True)

    return batch

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

def smart_model_load(model, pretrain_state_dict, loading_method='standard'):

    # Pass the model and the pretrained state_dict into the loading method
    eval(f'{loading_method}_model_load')(model, pretrain_state_dict)

    
def standard_model_load(model, pretrain_state_dict):

    # parse from multiple gpu to single or vice versa
    parsed_state_dict = {}
    for k, v in pretrain_state_dict.items():
        if k not in model.state_dict():
            if k.startswith('module.'):
                k = k[len('module.'):]
            else:
                k = 'module.' + k
        if k in model.state_dict():
            parsed_state_dict[k] = v

    # delete the linear classifier
    for k in model.state_dict():
        if k.startswith('module.linear') or k.startswith('linear'):
            del parsed_state_dict[k]

    # Now load this state dict to our model
    new_state_dict = model.state_dict()
    new_state_dict.update(parsed_state_dict)
    model.load_state_dict(new_state_dict)

def standard_with_classifier_model_load(model, pretrain_state_dict):

    # parse from multiple gpu to single or vice versa
    parsed_state_dict = {}
    for k, v in pretrain_state_dict.items():
        if k not in model.state_dict():
            if k.startswith('module.'):
                k = k[len('module.'):]
            else:
                k = 'module.' + k
        if k in model.state_dict():
            parsed_state_dict[k] = v

    # Now load this state dict to our model
    new_state_dict = model.state_dict()
    new_state_dict.update(parsed_state_dict)
    model.load_state_dict(new_state_dict)

def res50_from_res101_manual_model_load(model, pretrain_state_dict):

    # parse from multiple gpu to single or vice versa
    parsed_state_dict = {}
    for k, v in pretrain_state_dict.items():
        if k in model.state_dict():
            parsed_state_dict[k] = v

    # Define the dictionary for picking
    replacement = {
            '1': 10,
            '2': 19,
            '3': 20,
            '4': 21,
            '5': 22,
            }

    # Replacement function for layer3.* blocks
    for k in parsed_state_dict.keys():
        if k.startswith('layer3'):
            split = k.split('.')
            block_num = split[1]
            if block_num in replacement:
                split[1] = str(replacement[block_num])
                layer_name = '.'.join(split)
                assert layer_name in pretrain_state_dict, "Wrong replacement!!"
                parsed_state_dict[k] = pretrain_state_dict[layer_name]

    # Now load this state dict to our model
    new_state_dict = model.state_dict()
    new_state_dict.update(parsed_state_dict)
    model.load_state_dict(new_state_dict)

def additional_head_model_load(model, pretrain_state_dict):

    # considering single gpu only
    parsed_state_dict = {}
    for k, v in pretrain_state_dict.items():
        if k in model.state_dict():
            parsed_state_dict[k] = v

    # Define the dictionary for picking
    replacement = {
            '1': 1,
            '2': 2,
            '3': 3,
            '4': 4,
            '5': 5,
            }

    """
    We need to replace two things:
    1. Blocks in layer 3, based on the replacement dic above
    2. additional_fc
    """
    for k in parsed_state_dict.keys():
        if 'fc' in k:
            k_additional = k.replace('fc', 'additional_fc')
            assert k_additional in pretrain_state_dict, "Something's wrong, I can feel it!"
            parsed_state_dict[k] = pretrain_state_dict[k_additional]
        elif k.startswith('all_layers'):
            split = k.split('.')
            layer_num = k[1]
            block_num = k[2]
            if layer_num == 2:
                if block_num in replacement:
                    split[2] = str(replacement[block_num])
                    layer_name = '.'.join(split)
                    assert layer_name in pretrain_state_dict, "Wrong replacement!!"
                    parsed_state_dict[k] = pretrain_state_dict[layer_name]

    # Now load this state dict to our model
    new_state_dict = model.state_dict()
    new_state_dict.update(parsed_state_dict)
    model.load_state_dict(new_state_dict)

def parallel_model_load(model, pretrain_state_dict):

    # parse from multiple gpu to single or vice versa
    parsed_state_dict = {}
    for k, v in pretrain_state_dict.items():
        if k not in model.state_dict():
            if k.startswith('module.'):
                k = k[len('module.'):]
            else:
                k = 'module.' + k
        if k in model.state_dict():
            parsed_state_dict[k] = v

    # now load the same parameters in parallel blocks
    for k in model.state_dict():
        if 'parallel_' in str(k):
            parsed_state_dict[k] = parsed_state_dict[k.replace('parallel_', '')]

    # delete the linear classifier
    for k in model.state_dict():
        if k.startswith('module.linear') or k.startswith('linear'):
            del parsed_state_dict[k]

    # Now load this state dict to our model
    new_state_dict = model.state_dict()
    new_state_dict.update(parsed_state_dict)
    model.load_state_dict(new_state_dict)

