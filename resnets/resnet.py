#----------------------------------------
#--------- Torch Related Imports --------
#----------------------------------------
import torch
import torch.nn as nn

#----------------------------------------
#--------- Python Lib Imports -----------
#----------------------------------------
import math

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv5x5(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, bias=False)

class DownsampleB(nn.Module):
    
    def __init__(self, nIn, nOut, stride=2):
        super(DownsampleB, self).__init__()
        self.avg = nn.AvgPool2d(stride)

    def forward(self, x):
        residual = self.avg(x)
        return torch.cat((residual, residual*0), 1)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, conv_layer=conv3x3, downsample=None, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = conv_layer(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(True)
        self.conv2 = conv_layer(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, config):
        super(ResNet, self).__init__()

        """
        Factor: The number of after the first resnet block will be 32*factor
        """
        factor = 1

        # The pre-cursor to resnet blocks
        self.conv1 = conv3x3(3, int(32*factor))
        self.bn1 = nn.BatchNorm2d(int(32*factor))

        """
        The resnet backbone would be a nn.ModuleList() consisting of all blocks in all parts
        """
        # Create the resnet blocks backbone
        self.backbone = self._make_backbone(config, factor)

        """
        Other remaining layers
        The number of channels in the activation maps will increase to 256*factor,
        doubling each set of blocks and there will be three sets of blocks.
        And then adaptive pool to reduce n*n maps to 1*1
        """
        self.bn2 = nn.Sequential(nn.BatchNorm2d(int(256*factor)), nn.ReLU(inplace=True))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(int(256*factor), config.NUM_CLASS)

        
    def _make_backbone(self, config, factor=1):

        """
        The backbone will be a ModuleList consisting of multiple blocks in each part
        The strides and filt_sizes are hardcoded considering resnet like blocks only
        In future if you need to add backbones consisting of other kind of blocks,
        like that in GooLeNet, just write another _make_backbone function.
        """
        backbone = []

        self.in_planes = int(32*factor)
        strides = [2,2,2]
        planes = [64, 128, 256]

        # construct residual blocks for each part
        for idx, (num_planes, num_blocks, stride) in enumerate(zip(planes, config.LAYERS, strides)):
            blocks = self._make_layer(num_planes, num_blocks, block=eval(config.BLOCK), conv_layer=eval(config.CONV_LAYER), stride=stride)
            backbone += blocks

        # convert the list of blocks into a modulelist
        backbone = nn.ModuleList(backbone)

        return backbone

    def _make_layer(self, planes, num_blocks, block=BasicBlock, conv_layer=conv3x3, stride=1):

        # The first block has to have downsample layer
        """
        The first block is marked by the difference in the number of planes and in_planes
        """
        if self.in_planes != planes * block.expansion:
            downsample = DownsampleB(self.in_planes, planes * block.expansion, 2)
        else:
            downsample = None

        # The first block will have downsample and stride = 2
        blocks = [block(self.in_planes, planes, conv_layer=conv_layer, downsample=downsample, stride=stride)]

        # Now we can increment the number of in_planes
        self.in_planes = planes * block.expansion

        # Add rest of the blocks with stride 1
        for i in range(1, num_blocks):
            blocks.append(block(self.in_planes, planes))

        return blocks

    def seed(self, x):
        x = self.bn1(self.conv1(x))
        return x

    def top(self, x):
        x = self.bn2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def forward(self, x):

        # apply the initial conv + batchnorm layers
        x = self.seed(x)

        # Apply the backbone
        for block in self.backbone:
            x = block(x)

        # Apply the last layers
        x = self.top(x)

        return x



