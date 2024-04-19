import datetime
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
from torch.utils.tensorboard import SummaryWriter

from parse_args import parse_args, getModel

from dataset import MyDataSet
from utils import read_split_data, evaluate


# tensorboard --logdir=./runs --port=2000

def test(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            images_size=args.image_size)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # 如果存在预训练权重则载入
    model = getModel(args)
    model_weight_path = "./weights/{}_best_model.pth".format(args.arch)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    start_time = time.time()

    # validate
    val_loss, val_acc = evaluate(model=model,
                                 data_loader=val_loader,
                                 device=device,
                                 epoch=0)

    print("test loss:{:.4f} acc:{:.4f}".format(val_loss, val_acc * 100))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("test time {}".format(total_time_str))


if __name__ == '__main__':
    args = parse_args()

    test(args)
