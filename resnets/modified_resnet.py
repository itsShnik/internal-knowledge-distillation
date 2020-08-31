# -------------------------------------------
# Taken from https://github.com/gyhui14/spottune
# Modified by Nikhil shah
# -------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class DownsampleB(nn.Module):
    def __init__(self, nIn, nOut, stride=2):
        super(DownsampleB, self).__init__()
        self.avg = nn.AvgPool2d(stride)

    def forward(self, x):
        residual = self.avg(x)
        return torch.cat((residual, residual*0),1)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# No projection: identity shortcut
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Sequential(nn.ReLU(True), conv3x3(planes, planes))
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        y = self.bn2(out)

        return y

class ResNet(nn.Module):
    def __init__(self, block, layers, num_class = 10, training_strategy='standard'):
        super(ResNet, self).__init__()

        # self.training strategy
        self.training_strategy = training_strategy

        # initial layers before residual blocks
        factor = 1
        self.in_planes = int(32*factor)
        self.conv1 = conv3x3(3, int(32*factor))
        self.bn1 = nn.BatchNorm2d(int(32*factor))
        self.relu = nn.ReLU(inplace=True)

        # parameters for residual layers
        strides = [2, 2, 2]
        filt_sizes = [64, 128, 256]

        # lists for blocks and downsamples
        self.blocks, self.downsample = [], []

        # construct residual blocks
        for idx, (filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, layers, strides)):
            blocks, downsample = self._make_layer(block, filt_size, num_blocks, stride=stride)
            self.blocks.append(nn.ModuleList(blocks))
            self.downsample.append(downsample)

        # self.blocks: Multiple layers, each consisting of multiple residual blocks
        self.blocks = nn.ModuleList(self.blocks)
        self.downsample = nn.ModuleList(self.downsample)

        # initialize parallel blocks if the training strategy requires
        if self.training_strategy == 'SpotTune':
            self.parallel_blocks, self.parallel_downsample = [], []

            # construct parallel layers if training strategy requires
            for idx, (filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, layers, strides)):
                blocks, downsample = self._make_layer(block, filt_size, num_blocks, stride=stride)
                self.parallel_blocks.append(nn.ModuleList(blocks))
                self.parallel_downsample.append(downsample)

            # self.parallel_blocks: Same as self.blocks
            self.parallel_blocks = nn.ModuleList(self.parallel_blocks)
            self.parallel_downsample = nn.ModuleList(self.parallel_downsample)

            # Freezze the params of parallel blocks
            for params in self.parallel_blocks.parameters():
                params.requires_grad = False
            for params in self.parallel_downsample.parameters():
                params.requires_grad = False

        # remaining layers to project into required number of classes
        self.bn2 = nn.Sequential(nn.BatchNorm2d(int(256*factor)), nn.ReLU(True)) 
        self.avgpool = nn.AdaptiveAvgPool2d(1)        
        self.linear = nn.Linear(int(256*factor), num_class)

        # save the layers: List into the layer config
        self.layer_config = layers

        # Weight normalizations for Conv2d weights and zero initializations for batchnorm
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def seed(self, x):
        x = self.bn1(self.conv1(x))
        return x

    def _make_layer(self, block, planes, num_blocks, stride=1):

        # create a sequential downsample layer
        downsample = nn.Sequential()
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = DownsampleB(self.in_planes, planes * block.expansion, 2)

        # residual blocks for each layer
        blocks = [block(self.in_planes, planes, stride)]
        self.in_planes = planes * block.expansion
        for i in range(1, num_blocks):
            blocks.append(block(self.in_planes, planes))

        return blocks, downsample

    def forward(self, x, policy=None):

        # apply the initial conv + batchnorm layers
        x = self.seed(x)
    
        # Now apply the residual blocks based on the strategy
        if self.training_strategy == 'SpotTune':

            # current index specifies the index of policy that we are going to use
            current_index = 0

            for layer, num_blocks in enumerate(self.layer_config):
                    for block in range(num_blocks):
                        # calculate the residual and block ouput
                        residual = self.downsample[layer](x) if block==0 else x
                        output = self.blocks[layer][block](x)
                        output = F.relu(residual + output)
			
                        # calculate the parallel side outputs and residuals
                        parallel_residual = self.parallel_downsample[layer](x) if block==0 else x
                        parallel_output = self.parallel_blocks[layer][block](x)
                        parallel_output = F.relu(parallel_residual + parallel_output)

                        # take a decision here
                        action = policy[:,current_index].contiguous()
                        action_mask = action.float().view(-1,1,1,1)
                        x = action_mask * output + (1-action_mask) * parallel_output
                        current_index += 1   

        else:
            for layer, num_blocks in enumerate(self.layer_config):
                for block in range(num_blocks):
                    # calculate the results in the standard manner
                    residual = self.downsample[layer](x) if block==0 else x
                    output = self.blocks[layer][block](x)
                    x = F.relu(residual + output)

        x = self.bn2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
