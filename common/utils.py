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

def from_previous_layers_model_load(model, pretrain_state_dict):

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

    # Initialize the 3rd, 4th, 5th and maybe 6th block with 2nd block
    for k in model.state_dict():
        # The index at which block num is in k_split
        if k.startswith('module.'):
            ind = 2
        else:
            ind = 1
        # Check if it is not in parsed_state_dict
        if k not in parsed_state_dict:
            k_split = k.split('.')
            if int(k_split[ind]) >= 2:
                k_split[ind] = '1'
            modified_k = '.'.join(k_split)
            if modified_k in parsed_state_dict:
                parsed_state_dict[k] = parsed_state_dict[modified_k]
        else:
            continue

    # delete the linear classifier
    for k in model.state_dict():
        if k.startswith('module.linear') or k.startswith('linear'):
            del parsed_state_dict[k]

    # Now load this state dict to our model
    new_state_dict = model.state_dict()
    new_state_dict.update(parsed_state_dict)
    model.load_state_dict(new_state_dict)
