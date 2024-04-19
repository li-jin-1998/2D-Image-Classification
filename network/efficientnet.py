from collections import OrderedDict
from functools import partial
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchsummary import summary
from torchvision.models import efficientnet

nonlinearity = partial(F.relu, inplace=True)

n_channels_dict = {'efficientnet_b0': 1280, 'efficientnet_b1': 1280, 'efficientnet_b2': 1408,
                   'efficientnet_b3': 1536, 'efficientnet_b4': 1792, 'efficientnet_b5': 2048,
                   'efficientnet_b6': 2304, 'efficientnet_b7': 2560}


class IntermediateLayerGetter(nn.ModuleDict):
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class EfficientNet(nn.Module):
    def __init__(self, num_classes, pretrain_backbone: bool = True, model_name: str = None):
        super(EfficientNet, self).__init__()
        print(model_name)
        if model_name == 'efficientnet_b0':
            backbone = efficientnet.efficientnet_b0(pretrained=pretrain_backbone)
        elif model_name == 'efficientnet_b1':
            backbone = efficientnet.efficientnet_b1(pretrained=pretrain_backbone)
        elif model_name == 'efficientnet_b2':
            backbone = efficientnet.efficientnet_b2(pretrained=pretrain_backbone)
        elif model_name == 'efficientnet_b3':
            backbone = efficientnet.efficientnet_b3(pretrained=pretrain_backbone)
        elif model_name == 'efficientnet_b4':
            backbone = efficientnet.efficientnet_b4(pretrained=pretrain_backbone)
        elif model_name == 'efficientnet_b5':
            backbone = efficientnet.efficientnet_b5(pretrained=pretrain_backbone)
        elif model_name == 'efficientnet_b6':
            backbone = efficientnet.efficientnet_b6(pretrained=pretrain_backbone)
        elif model_name == 'efficientnet_b7':
            backbone = efficientnet.efficientnet_b7(pretrained=pretrain_backbone)
        else:
            exit(1)

        backbone = backbone.features
        stage_indices = [8]
        # self.stage_out_channels = [backbone[i].out_channels for i in stage_indices]
        return_layers = dict([(str(j), f"stage{i}") for i, j in enumerate(stage_indices)])
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            # nn.Dropout(0.5),
            # nn.Linear(n_channels_dict[model_name], 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(n_channels_dict[model_name], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)  # rgb
        backbone_out = self.backbone(x)
        x = backbone_out['stage0']
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = F.sigmoid(x)
        return x


if __name__ == '__main__':
    model = EfficientNet(num_classes=2, pretrain_backbone=True, model_name='efficientnet_b1').to('cuda')
    summary(model, (3, 192, 192))
