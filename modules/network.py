#----------------------------------------
#--------- Resnet Imports ---------------
#----------------------------------------
from resnets.resnet import ResNet

def resnet(config):

    """
    Just return a resnet object and pass config.MAIN
    """
    return ResNet(config.MAIN)

def dynamic_resnet(config):

    """
    Just return a dynamic_resnet object and pass config
    """
    pass
