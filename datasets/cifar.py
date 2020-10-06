#----------------------------------------
#--------- OS related imports -----------
#----------------------------------------
import os

#----------------------------------------
#--------- Torch related imports --------
#----------------------------------------
import torchvision.transforms as transforms
import torchvision.datasets as datasets


# We can make either a function or a class for CIFAR datasets, here we go for function
def cifar10(root='data/cifar-10', splits='train', toy=False):

    # create dataset object and return
    if splits in ['train']:
        dataset = datasets.CIFAR10(root=root,
                train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
                ]))

    else:
        dataset = datasets.CIFAR10(root=root,
                train=False,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
                ]))

    return dataset

def cifar100(root='data/cifar-100', splits='train', toy=False):

    # create dataset object and return
    if splits in ['train']:
        dataset = datasets.CIFAR100(root=root,
                train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
                ]))

    else:
        dataset = datasets.CIFAR100(root=root,
                train=False,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
                ]))

    return dataset
