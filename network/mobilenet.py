from collections import OrderedDict
from functools import partial
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchsummary import summary
from torchvision.models import mobilenet_v3_large

nonlinearity = partial(F.relu, inplace=True)


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


class MobileNet(nn.Module):
    def __init__(self, num_classes, pretrain_backbone: bool = True):
        super(MobileNet, self).__init__()
        backbone = mobilenet_v3_large(pretrained=pretrain_backbone)

        backbone = backbone.features

        stage_indices = [1, 3, 6, 12, 15]
        # self.stage_out_channels = [backbone[i].out_channels for i in stage_indices]
        return_layers = dict([(str(j), f"stage{i}") for i, j in enumerate(stage_indices)])
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        self.fc = nn.Linear(160, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[-2:]
        backbone_out = self.backbone(x)
        # encoder
        # e0 = backbone_out['stage0']
        # e1 = backbone_out['stage1']
        # e2 = backbone_out['stage2']
        # e3 = backbone_out['stage3']
        e4 = backbone_out['stage4']
        x = self.avgpool(e4)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.sigmoid(x)
        return x


if __name__ == '__main__':
    model = MobileNet(num_classes=2).to('cuda')
    summary(model, (3, 192, 192))
