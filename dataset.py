import random

import PIL.Image
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

torch.manual_seed(2023)


def standardize_image(image, mean=None,
                      std=None):
    image = image / 255
    # 分别标准化每个通道
    if mean is None:
        mean = [0.66439311, 0.47282529, 0.40711038]
    if std is None:
        std = [0.1270305, 0.13880533, 0.12055053]
    standardized_image = np.zeros_like(image)
    for i in range(image.shape[2]):
        channel = image[:, :, i]
        standardized_channel = (channel - mean[i]) / std[i]
        standardized_image[:, :, i] = standardized_channel
    print(np.max(standardized_image))
    return standardized_image


def preprocessing(image, image_size):
    image = image.resize((image_size, image_size),
                         PIL.Image.BILINEAR)
    image = np.array(image, np.float32)
    image = image / 127.5 - 1
    # image=standardize_image(image)
    # image = image / 255
    return image


def pre_process(path, image_size, transform=False):
    img = cv2.imread(path)
    if img is None:
        print(path)
        pass
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    if transform:
        scale = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        img = cv2.convertScaleAbs(img, alpha=scale, beta=contrast)
        # print(scale,contrast)
    # img = cv2.bilateralFilter(img, 2, 50, 50)  # remove images noise.
    # img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)  # produce a pseudocolored image. 伪彩色
    img = np.array(img, np.float32)
    img = img / 127.5 - 1
    return img


class MyDataSet(Dataset):
    def __init__(self, images_paths: list, images_class: list, images_size: int, transform=None):
        self.images_paths = images_paths
        self.images_class = images_class
        self.images_size = images_size
        self.transform = transform

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, index):
        label = self.images_class[index]
        # img=self.images[index]
        img = torch.Tensor(pre_process(self.images_paths[index], self.images_size, transform=self.transform))
        img = img.permute(2, 0, 1)
        # if self.transform is not None:
        #     img = self.transform(img)
        # print(img.shape)
        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
