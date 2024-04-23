import argparse

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

efficientnet_dict = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                     'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
                     'efficientnet_b6', 'efficientnet_b7']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', '-a', metavar='ARCH', default='efficientnet_b1',
                        help='/mobilenet/mobilevit/efficientnet')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument("--image_size", default=192, type=int)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--resume', default=False, help='resume from checkpoint')
    parser.add_argument('--data_path', type=str,
                        default="/mnt/algo_storage_server/ScanSceneClassification")
    parser.add_argument('--freeze_layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')

    return parser.parse_args()


def get_model(args):
    print('**************************')
    print(f'model:{args.arch}\n'
          f'epoch:{args.epochs}\n'
          f'batch size:{args.batch_size}\n'
          f'image size:{args.image_size}')
    print('**************************')

    if args.arch == 'mobilenet':
        from network.mobilenet import MobileNet
        model = MobileNet(num_classes=args.num_classes).to(device)

    if args.arch == 'mobilevit':
        from MobileViT.mobilevit import mobile_vit_small
        model = mobile_vit_small(num_classes=args.num_classes).to(device)

    if args.arch == 'efficientnet' or args.arch in efficientnet_dict:
        # args.arch = 'efficientnet_b1'
        from network.efficientnet import EfficientNet
        model = EfficientNet(num_classes=args.num_classes, pretrain_backbone=True,
                             model_name=args.arch).to(device)

    if args.arch == 'efficientnet2':
        from efficientunet import EfficientNet
        model = EfficientNet.custom_head(model_name='efficientnet-b1',
                                         n_classes=args.num_classes, pretrained=True).to(device)

    return model
