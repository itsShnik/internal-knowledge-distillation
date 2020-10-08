#----------------------------------------
#--------- Torch related imports --------
#----------------------------------------
import torch

def to_cuda(batch):
    # convert batch: tuple to batch: list
    batch = list(batch)

    for i in range(len(batch)):
        assert isinstance(batch[i], torch.Tensor), "Each element of batch is not a tensor"
        batch[i] = batch[i].cuda(non_blocking=True)

    return batch

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

