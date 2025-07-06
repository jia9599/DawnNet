# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from segment_anything.utils.transforms import ResizeLongestSide



def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def tensor_to_image(tensor):
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.permute(1, 2, 0)
    tensor = tensor.mul(255).clamp(0, 255)
    tensor = tensor.cpu().numpy().astype('uint8')
    return tensor

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y, _ = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32))
        image = image.permute(2, 0, 1)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

class Resize(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        padding = (self.output_size - 512)//2
        Pad_PIL = transforms.Pad(padding, fill=(0, 0, 0), padding_mode='constant')
        img_pil = Image.fromarray(image)
        image = Pad_PIL(img_pil)
        image = np.asarray(image)

        Pad_PIL_label = transforms.Pad(padding, fill=0, padding_mode='constant')
        label_pil = Image.fromarray(label)
        label = Pad_PIL_label(label_pil)
        label = np.asarray(label)

        image = torch.from_numpy(image.astype(np.float32))
        image = image.permute(2, 0, 1)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample



class MMIS_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, fold=None):
        self.fold = fold
        self.split = split
        # Whether to use 5-fold cross-validated dataset
        if self.fold != None:
            self.sample_list = open(os.path.join(list_dir, '5_fold', self.split + str(self.fold) +'.txt')).readlines()
        else:
            self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir
        self.sam_trans = ResizeLongestSide(1024)

        self.transform = {
            "image": transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((512, 512), transforms.InterpolationMode.BILINEAR),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            "label": transforms.Compose([
                transforms.Resize((512, 512), transforms.InterpolationMode.NEAREST),
            ])
        }


    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            if self.fold != None:
                data_path = os.path.join(self.data_dir, '5_fold_train_val', slice_name+ '.npz')
            else:
                data_path = os.path.join(self.data_dir, 'train_TC_npz', slice_name + '.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            label = torch.from_numpy(label.astype(np.float32))
            label = label.unsqueeze(dim=0)
        elif self.split == "val":
            slice_name = self.sample_list[idx].strip('\n')
            if self.fold != None:
                data_path = os.path.join(self.data_dir, '5_fold_train_val', slice_name+'.npz')
            else:
                data_path = os.path.join(self.data_dir, 'val_TC_npz', slice_name + '.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            label = torch.from_numpy(label.astype(np.float32))
            label = label.unsqueeze(dim=0)
        elif self.split == "test":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, 'test_TC_npz', slice_name + '.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            label = torch.from_numpy(label.astype(np.float32))
            label = label.unsqueeze(dim=0)
        if self.transform:
            image = self.transform["image"](image)
            label = self.transform["label"](label)

        sample = {'image': image, 'label': label}

        sample['case_name'] = self.sample_list[idx].strip('\n')
        sample['original_size'] = (512, 512)
        sample['mask_inputs'] = None
        return sample

def collate_fn(batch):
    batch_size = len(batch)
    return batch

















