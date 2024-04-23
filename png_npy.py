import os
import random

import cv2
import numpy as np
import tqdm


def pre_process(path, image_size):
    img = cv2.imread(path)
    if img is None:
        print(path)
        pass
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    if 'train' in path:
        scale = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        img = cv2.convertScaleAbs(img, alpha=scale, beta=contrast)
    img = np.array(img, np.float32)
    img = img / 127.5 - 1
    return img


def png_npy(image_path, image_size):
    images = []
    for path in tqdm.tqdm(image_path):
        image = pre_process(path, image_size)
        images.append(image)
    return np.array(images)


if __name__ == '__main__':
    from parse_args import parse_args
    from utils import read_split_data

    args = parse_args()
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
    # print(train_images_path, train_images_label)
    train_images = png_npy(train_images_path, args.image_size)
    np.save(os.path.join(args.data_path, 'train_images.npy'), train_images)
    np.save(os.path.join(args.data_path, 'train_labels.npy'), train_images_label)
    val_images = png_npy(val_images_path, args.image_size)
    np.save(os.path.join(args.data_path, 'val_images.npy'), val_images)
    np.save(os.path.join(args.data_path, 'val_labels.npy'), val_images_label)
    # images = np.load(os.path.join(args.data_path, 'train_images.npy'))
    # print(images.shape)
