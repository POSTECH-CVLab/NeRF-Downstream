import gin
import torch
import torch.nn as nn
import timm

import torchvision.models as models

@gin.configurable()
class ResNetBased(nn.Module):
    def __init__(
        self, 
        model: str = "resnet18",
        dropout_rate = 0.2,
        pretrained: bool = False,
    ):
        super(ResNetBased, self).__init__()
        model = torch.hub.load(
            'pytorch/vision:v0.10.0', 
            model, 
            pretrained=pretrained,
            zero_init_residual=True,
        )
        in_features = model.fc.in_features
        model.fc = nn.Identity()
        self.fc = nn.Linear(
            in_features=in_features,
            out_features=51,
            bias=True
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.model = model
    
    def forward(self, x):
        return self.fc(self.dropout(self.model(x)))


@gin.configurable()
class ViTBased(nn.Module):

    def __init__(
        self, 
        model: str = "vit_small_patch16_224",
        pretrained: bool = False
    ):
        super(ViTBased, self).__init__()
        model = timm.create_model(
            model,
            pretrained=pretrained,
            num_classes=51
        )
        self.model = model

    def forward(self, x):
        return self.model(x)
