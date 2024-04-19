import datetime
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import math

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
from parse_args import parse_args, getModel

from dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate


# tensorboard --logdir=./runs --port=2000

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:2000/')
    tb_writer = SummaryWriter()
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              images_size=args.image_size)

    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            images_size=args.image_size)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = getModel(args)
    if args.resume:
        weights_path = "./weights/{}_best_model.pth".format(args.arch)
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # print(name)
            for i in range(0, 6):
                if name.startswith("backbone." + str(i)):
                    para.requires_grad_(False)
                    print("freeze {}".format(name))
                    continue
                # else:
                #     print("training {}".format(name))
    # summary(model, (3, args.image_size, args.image_size))
    # exit(0)
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adamax(params_to_optimize, lr=args.lr)
    # optimizer = torch.optim.SGD(params_to_optimize, lr=args.lr, momentum=0.9, weight_decay=1E-4)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf, verbose=True)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, last_epoch=-1, gamma=0.95, verbose=True)
    acc = 0
    best_epoch = 1
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), "./weights/{}_latest_model.pth".format(args.arch))
        if acc <= val_acc:
            best_epoch = epoch
            acc = val_acc
            torch.save(model.state_dict(), "./weights/{}_best_model.pth".format(args.arch))
        print("best epoch:{} acc:{:.4f}".format(best_epoch, acc * 100))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


if __name__ == '__main__':
    print('>' * 10, 'train')
    args = parse_args()

    train(args)
