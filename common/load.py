#----------------------------------------
#--------- Torch related imports --------
#----------------------------------------


def smart_model_load(model, pretrain_state_dict):
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

    # Now load this state dict to our model
    new_state_dict = model.state_dict()
    new_state_dict.update(parsed_state_dict)
    model.load_state_dict(new_state_dict)
