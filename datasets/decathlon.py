#----------------------------------------
#--------- Torch related imports --------
#----------------------------------------
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data

#----------------------------------------
#--------- OS and tools -----------------
#----------------------------------------
import os
import os.path
from pathlib import Path
import numpy as np
import pickle
from PIL import Image
from pycocotools.coco import COCO

# Image loader from path
def pil_loader(path):
    return Image.open(path).convert('RGB')

# Annotation file loader using coco
def load_from_annotation_files(root, annotation_files):
    image_paths, labels = [], []

    for annFile in annotation_files:
        coco = COCO(annFile)

        # load images and annotations
        imgIds = coco.getImgIds()
        images = coco.loadImgs(imgIds)

        annIds = coco.getAnnIds(imgIds=imgIds)
        anno = coco.loadAnns(annIds)

        """
        relative_image_path[0] for imagenet12_train:
        'data/imagenet12/train/0001/000001.jpg'
        """
        relative_image_paths = [img['file_name'] for img in images]
        image_ids = [img['id'] for img in images]

        # labels should start from 0
        temp_labels = [int(ann['category_id'])-1 for ann in anno]
        labels += temp_labels

        # Create absolute paths for images
        """
        image_paths[0] for imagenet12_train:
        ('/path/to/root/data/imagenet12/train/0001/000001.jpg', '<id of the image>')
        """

        for j in range(len(relative_image_paths)):
            image_paths.append((os.path.join(root, relative_image_paths[j]), image_ids[j]))

    # labels should start from 0
    min_lab = min(labels)
    labels = [lab - min_lab for lab in labels]

    return image_paths, labels

def create_transforms(name):
    # Load the means and stds file
    dict_mean_std = pickle.load(open(os.path.join(root, 'decathlon_mean_std.pickle'), 'rb'), encoding='latin1')
    means = dict_mean_std[name + 'mean']
    stds = dict_mean_std[name + 'std']

    if name in ['gtsrb', 'omniglot', 'svhn']:
        transform = transforms.Compose([
            transforms.Resize(72),
            transforms.CenterCrop(72),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(72),
            transforms.RandomCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])

    return transform

# Functional interfaces for all the datasets in Decathlon
def decathlon_dataset(name='imagenet12', root='data/decathlon', splits='train+val'):

    splits = splits.split('+')

    cache_file_name = '_'.join([name] + splits) + '.pkl'
    cache_dir = 'cache'
    cache_file_path = os.path.join(cache_dir, cache_folder_name)

    if os.path.exists(cache_file_path):
        print('Loading dataset from cache!!')
        image_paths, labels = pickle.load(open(cache_file_path, 'rb'))

    else:
        # annotation files
        annotation_files = []
        for split in splits.split('+'):
            if split in ['train', 'val']:
                annotation_files.append(os.path.join(root, 'annotations', f'{name}_{split}.json'))
            else:
                annotation_files.append(os.path.join(root, 'annotations', f'{name}_stripped_test.json'))

        # load all images and labels from given annotation files
        image_paths, labels = load_from_annotation_files(root, annotation_files)

        # cache the image_paths and labels
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        pickle.dump((image_paths, labels), open(cache_file_path, 'wb'))

    # create transforms
    transforms = create_transforms(name)

    # return an ImageFolder object
    return ImageFolder(root, transform=transforms, target_transform=None, index=None, labels=labels, imgs=image_paths)

# A general class for all Decathlon datasets
class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None, index=None,
            labels=None ,imgs=None,loader=pil_loader,skip_label_indexing=0):
        
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"))

        self.root = root
        if index is not None:
            imgs = [imgs[i] for i in index]
        self.imgs = imgs
        if index is not None:
            if skip_label_indexing == 0:
                labels = [labels[i] for i in index]
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index][0]
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

