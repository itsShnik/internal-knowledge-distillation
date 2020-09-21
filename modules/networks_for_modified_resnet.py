#----------------------------------------
#--------- Resnet Imports ---------------
#----------------------------------------
from resnets.modified_resnet import ResNet, DynamicResNet

def modified_resnet(config=None):

    """
    Just return a resnet object and pass config.MAIN
    """
    return ResNet(config.MAIN)

def modified_dynamic_resnet(config=None):

    """
    Just return a dynamic_resnet object and pass config
    """
    return DynamicResNet(config)
