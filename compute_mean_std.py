import os

import numpy as np
from PIL import Image


def main():
    img_channels = 3
    img_dir = "/home/lij/PycharmProjects/Seg/data/train/image"
    roi_dir = "/home/lij/PycharmProjects/Seg/data/train/mask"
    assert os.path.exists(img_dir), f"image dir: '{img_dir}' does not exist."
    assert os.path.exists(roi_dir), f"roi dir: '{roi_dir}' does not exist."

    img_name_list = [i for i in os.listdir(img_dir) if i.endswith(".png")]
    # print(img_name_list)
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)
    for img_name in img_name_list:
        img_path = os.path.join(img_dir, img_name)
        ori_path = os.path.join(roi_dir, img_name)
        img = np.array(Image.open(img_path)) / 255.
        roi_img = np.array(Image.open(ori_path))

        img = img[roi_img >= 0]
        cumulative_mean += img.mean(axis=0)
        cumulative_std += img.std(axis=0)
        # print(img_name,img.mean(axis=0))

    mean = cumulative_mean / len(img_name_list)
    std = cumulative_std / len(img_name_list)
    print(f"mean: {mean}")
    print(f"std: {std}")
    # mean: [0.66439311 0.47282529 0.40711038]
    # std: [0.1270305  0.13880533 0.12055053]


if __name__ == '__main__':
    main()
