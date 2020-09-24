#----------------------------------------
#--------- OS related imports -----------
#----------------------------------------
import os

#----------------------------------------
#--------- Torch related imports --------
#----------------------------------------
import torchvision.transforms as transforms
import torchvision.datasets as datasets


# We can make either a function or a class for Imagenet dataset, here we go for function
def imagenet(root='data/Imagenet/', splits='train', toy=False):

    # Set up transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # data dir
    data_dir = os.path.join(root, splits)

    # create dataset object and return
    if splits in ['train']:
        dataset = datasets.ImageFolder(
            data_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))


    else:
        dataset = datasets.ImageFolder(
            data_dir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

    return dataset
