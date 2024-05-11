from typing import Iterator, Tuple

import timm
import torch
from torch import nn
from torch.nn import Parameter


class AlexNet(nn.Module):

    def __init__(self, categories, is_rgb, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        input_channels = 3 if is_rgb else 1
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 96, 11, 4),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            nn.Flatten(),
            nn.Linear(9216, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, out_features=categories)
        )

    def forward(self, input):
        return self.model(input)


class ResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1, use_1x1conv=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, output_channels, 1, stride=stride)  # 这里stride2其实不太合理
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        save = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.conv3:
            save = self.conv3(save)

        x += save
        x = torch.nn.functional.relu(x)
        return x


class ResNet18(nn.Module):

    def __init__(self, categories, is_rgb, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        input_channels = 3 if is_rgb else 1
        stage1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=7, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(3, 2, 1)
        )
        stage2 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64)
        )
        stage3 = nn.Sequential(
            ResBlock(64, 128, 2, True),
            ResBlock(128, 128)
        )
        stage4 = nn.Sequential(
            ResBlock(128, 256, 2, True),
            ResBlock(256, 256)
        )
        stage5 = nn.Sequential(
            ResBlock(256, 512, 2, True),
            ResBlock(512, 512)
        )
        self.model = nn.Sequential(
            stage1,
            stage2,
            stage3,
            stage4,
            stage5,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, categories)
        )

    def forward(self, x):
        x = self.model(x)
        return x


class PreTrainedResNet18(nn.Module):
    def __init__(self, categories, is_rgb, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = timm.create_model(model_name='resnet18', pretrained=True, num_classes=categories)

    def forward(self, x):
        x = self.model(x)
        return x


class PreTrainedResNet50(nn.Module):
    def __init__(self, categories, is_rgb, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = timm.create_model(model_name='resnet50', pretrained=True, num_classes=categories)

    def forward(self, x):
        x = self.model(x)
        return x
