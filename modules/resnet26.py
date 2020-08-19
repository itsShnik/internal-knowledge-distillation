from resnets.modified_resnet import BasicBlock, ResNet

# Just create a function for resnet26
def resnet26(num_class=10, blocks=BasicBlock):

    # layer configuration
    """
    3 types of layers, each consisting of 4 blocks
    First block type: BasicBlock with filter size : 64
    Second block type: BasicBlock with filter size : 128
    Third block type: BasicBlock with filter size : 256
    """

    layers = [4,4,4]

    return ResNet(blocks, layers, num_class)
