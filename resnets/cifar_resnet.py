"""
To reduce overfitting on CIFAR-100 dataset, changed the initial conv layers
"""
import torch
import torch.nn as nn

#----------------------------------------
#--------- Common imports ---------------
#----------------------------------------
import math
import sys
from common.utils.masks import *


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

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


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, drop_block=False, drop_rate=0.5):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            drop_block = False

        if drop_block:
            decision = torch.bernoulli(torch.tensor(drop_rate)).cuda()
            out = decision * out + identity
        else:
            out += identity

        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, training_strategy ='standard', num_additional_heads=1,
                 additional_mask_functions=None, drop_rate=0.5, classifier_first_dropout=0.1,
                 classifier_last_dropout=0.5, classifier_hidden_size=256, **kwargs):
        super(ResNet, self).__init__()
        self.training_strategy = training_strategy
        self.num_additional_heads = num_additional_heads
        self.drop_rate = drop_rate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # Initial conv layer slightly changed for CIFAR dataset
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.all_layers = nn.ModuleList([self.layer1, self.layer2, self.layer3, self.layer4])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # replace the fully connected layer with a stronger classifier
        #self.fc = nn.Linear(512 * block.expansion, num_classes)
        dim = 512*block.expansion
        self.final_mlp = torch.nn.Sequential(
                torch.nn.Dropout(classifier_first_dropout, inplace=False),
                torch.nn.Linear(dim, classifier_hidden_size),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(classifier_last_dropout, inplace=False),
                torch.nn.Linear(classifier_hidden_size, num_classes),
            )

        if self.training_strategy in ['AdditionalStochastic']:

            # Initialize an end of the network for the additional branch

            self.additional_avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.additional_mlp = torch.nn.Sequential(
                torch.nn.Dropout(classifier_first_dropout, inplace=False),
                torch.nn.Linear(dim, classifier_hidden_size),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(classifier_last_dropout, inplace=False),
                torch.nn.Linear(classifier_hidden_size, num_classes),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.ModuleList(layers)

    def train_forward(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self.training_strategy == 'AdditionalHeads':
            print("Change the branch for the Additional Heads strategy!")
            raise NotImplementedError
            
        elif self.training_strategy == 'Stochastic':
            # Calculate outputs for main branch and then for the other branch
            for layer in self.all_layers:
                for block in layer:
                    x = block(x, drop_block=True, drop_rate=self.drop_rate)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.final_mlp(x)

            return x

        elif self.training_strategy == 'AdditionalStochastic':

            return_list = []

            # First calculate the additional branches
            additional_x_output = x.clone()
            for layer_ind, layer in enumerate(self.all_layers):
                for block_ind, block in enumerate(layer):
                        additional_x_output = block(additional_x_output, drop_block=True, drop_rate=self.drop_rate)

            additional_x_output = self.additional_avgpool(additional_x_output)
            additional_x_output = torch.flatten(additional_x_output, 1)
            additional_x_output = self.additional_mlp(additional_x_output)

            return_list.append(additional_x_output)

            # Calculate outputs for main branch and then for the other branch
            for layer in self.all_layers:
                for block in layer:
                    x = block(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.final_mlp(x)

            return x, return_list

        else:

            for layer in self.all_layers:
                for block in layer:
                    x = block(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.final_mlp(x)

            return x


    def val_forward(self, x):

        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # At validation we can have as many strategies to prune/use blocks as we want
        # Currently there are five strategies
        # 1. Using the full model
        # 2. Prune every second block
        # 3. Prune every third block
        # 4. Prune based on the drop_rate
        # 5. Ensemble of all these

        return_list = []

        # Using the full model
        full_x = x.clone()
        for layer in self.all_layers:
            for block in layer:
                full_x = block(full_x)

        full_x = self.avgpool(full_x)
        full_x = torch.flatten(full_x, 1)
        full_x = self.final_mlp(full_x)

        # Pruned to use 50% of the blocks
        pruned_half_x = x.clone()
        block_count = 0
        for layer in self.all_layers:
            for block in layer:
                if block_count % 2 != 0:
                    pruned_half_x = block(pruned_half_x)
                else:
                    pruned_half_x = block(pruned_half_x, drop_block=True, drop_rate=0.0)
                block_count += 1

        pruned_half_x = self.avgpool(pruned_half_x)
        pruned_half_x = torch.flatten(pruned_half_x, 1)
        if self.training_strategy == 'AdditionalStochastic':
            pruned_half_x = self.additional_mlp(pruned_half_x)
        else:
            pruned_half_x = self.final_mlp(pruned_half_x)

        return_list.append(pruned_half_x)

        # Pruned to use only 1/3rd of the blocks
        pruned_third_x = x.clone()
        block_count = 0
        for layer in self.all_layers:
            for block in layer:
                if block_count % 3 != 0:
                    pruned_third_x = block(pruned_third_x)
                else:
                    pruned_third_x = block(pruned_third_x, drop_block=True, drop_rate=0.0)
                block_count += 1

        pruned_third_x = self.avgpool(pruned_third_x)
        pruned_third_x = torch.flatten(pruned_third_x, 1)
        if self.training_strategy == 'AdditionalStochastic':
            pruned_third_x = self.additional_mlp(pruned_third_x)
        else:
            pruned_third_x = self.final_mlp(pruned_third_x)

        return_list.append(pruned_third_x)

        # Pruned based on the drop rate
        pruned_drop_rate_x = x.clone()
        block_count = 0
        for layer in self.all_layers:
            for block in layer:
                if block_count % math.floor(1/self.drop_rate) != 0:
                    pruned_drop_rate_x = block(pruned_drop_rate_x)
                else:
                    pruned_drop_rate_x = block(pruned_drop_rate_x, drop_block=True, drop_rate=0.0)
                block_count += 1

        pruned_drop_rate_x = self.avgpool(pruned_drop_rate_x)
        pruned_drop_rate_x = torch.flatten(pruned_drop_rate_x, 1)
        if self.training_strategy == 'AdditionalStochastic':
            pruned_drop_rate_x = self.additional_mlp(pruned_drop_rate_x)
        else:
            pruned_drop_rate_x = self.final_mlp(pruned_drop_rate_x)

        return_list.append(pruned_drop_rate_x)

        # Take the sum of all the logits to calculate the ensemble
        ensemble_x = full_x + pruned_half_x + pruned_third_x + pruned_drop_rate_x
        return_list.append(ensemble_x)

        return full_x, return_list


    def forward(self, x, mode='train'):

        return eval(f'self.{mode}_forward')(x)
        
