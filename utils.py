import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
import torch
from torch.nn.functional import cross_entropy
from tqdm import tqdm

from dataset import MyDataSet
from parse_args import parse_args


def read_split_data(root: str):
    # random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    # classes = [cla for cla in os.listdir(os.path.join(root, 'train')) if
    #            os.path.isdir(os.path.join(root, 'train', cla))]
    classes = ["extra", "intra"]
    # 排序，保证各平台顺序一致
    classes.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(classes))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = {}  # 存储每个类别的样本总数
    # 遍历每个文件夹下的文件
    for cla in classes:
        txt_file = os.path.join(root, 'train_' + cla + '.txt')
        images = sorted([os.path.join(os.path.dirname(txt_file), cla, line.strip()) for line in open(txt_file)])
        # print(images)
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num['train_' + str(cla)] = len(images)
        for img_path in images:
            train_images_path.append(img_path)
            train_images_label.append(image_class)
    for cla in classes:
        txt_file = os.path.join(root, 'test_' + cla + '.txt')
        images = sorted([os.path.join(os.path.dirname(txt_file), cla, line.strip()) for line in open(txt_file)])
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num['test_' + str(cla)] = len(images)
        for img_path in images:
            val_images_path.append(img_path)
            val_images_label.append(image_class)
    for key in every_class_num:
        print("{}:{} images were found in the dataset.".format(key, every_class_num[key]))
    print("{} images for training and validation.".format(sum(every_class_num.values())))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(every_class_num)), every_class_num.values(), width=0.5, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(every_class_num)), every_class_num.keys())
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=every_class_num[v] + 6, s=str(every_class_num[v]), ha='center')
        plt.xlabel('image class')
        plt.ylabel('number of images')
        plt.title('class distribution')
        plt.tight_layout()
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 15)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        imgs = []
        labs = []
        for i in range(15):
            img = images[i].numpy().transpose(1, 2, 0)
            img = (img + 1) * 127.5
            label = labels[i].item()
            labs.append(label)
            imgs.append(img)
        # if len(imgs)==15:
        #     break

        fig, axes = plt.subplots(3, 5, figsize=(10, 10))
        axes = axes.flatten()
        for img, lab, ax in zip(imgs, labs, axes):
            ax.imshow(img.astype('uint8'), cmap=plt.get_cmap('gray'))
            ax.set_xlabel(class_indices[str(lab)])
            ax.set_xticks([])  # 去掉x轴的刻度
            ax.set_yticks([])  # 去掉y轴的刻度
        plt.tight_layout()
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def criterion(inputs, target):
    loss_weight = torch.as_tensor([1.0, 1.0], device="cuda")
    return cross_entropy(inputs, target, weight=loss_weight, label_smoothing=0.)


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = criterion(pred, labels.to(device))
        # b = 0.3
        # loss = (loss - b).abs() + b
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.4f}, acc: {:.4f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num * 100)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    model.eval()

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    with torch.no_grad():
        for step, data in enumerate(data_loader):
            images, labels = data
            sample_num += images.shape[0]

            pred = model(images.to(device))
            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels.to(device)).sum()

            loss = criterion(pred, labels.to(device))
            accu_loss += loss

            data_loader.desc = "[valid epoch {}] loss: {:.4f}, acc: {:.4f}".format(epoch,
                                                                                   accu_loss.item() / (step + 1),
                                                                                   accu_num.item() / sample_num * 100)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


if __name__ == '__main__':
    args = parse_args()
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    # 实例化验证数据集
    val_dataset = MyDataSet(images_paths=val_images_path,
                            images_class=val_images_label, images_size=192)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=16,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=8,
                                             collate_fn=val_dataset.collate_fn)
    plot_data_loader_image(val_loader)
