from resnets.modified_resnet import BasicBlock, ResNet

# A 8 layered resnet for policy network
def resnet8(num_class=24, blocks=BasicBlock):

    """
    num_class = 2 * total number of decisions to be made
    For main network == resnet26, num_class = 2 * 12 = 24
    """

    # layer configuration
    """
    3 types of layers, each consisting of 1 block
    First block type: BasicBlock with filter size : 64
    Second block type: BasicBlock with filter size : 128
    Third block type: BasicBlock with filter size : 256
    """

    layers = [1,1,1]

    return ResNet(blocks, layers, num_class, training_strategy='standard')
