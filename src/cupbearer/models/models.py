import math

import torch.nn.functional as F
from torch import nn

from .hooked_model import HookedModel


class MLP(HookedModel):
    def __init__(
        self,
        input_shape: list[int] | tuple[int, ...],
        output_dim: int,
        hidden_dims: list[int],
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        in_features = math.prod(input_shape)
        self.layers = nn.ModuleDict()
        for i_layer, (in_features, out_features) in enumerate(
            zip([in_features] + hidden_dims, hidden_dims + [output_dim])
        ):
            self.layers[f"linear_{i_layer}"] = nn.Linear(
                in_features=in_features,
                out_features=out_features,
            )
            self.layers[f"relu_{i_layer}"] = nn.ReLU()
        self.layers.pop(next(reversed(self.layers.keys())))  # rm last relu

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        for name, layer in self.layers.items():
            x = layer(x)
            self.store(f"post_{name}", x)
        return x

    @property
    def default_names(self) -> list[str]:
        return [
            f"post_{name}" for name in self.layers.keys() if name.startswith("linear")
        ]


class CNN(HookedModel):
    def __init__(
        self,
        input_shape: list[int] | tuple[int, ...],
        output_dim: int,
        channels: list[int],
        dense_dims: list[int],
        kernel_sizes: list[int] | None = None,
        strides: list[int] | None = None,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.channels = channels
        self.dense_dims = dense_dims
        self.kernel_sizes = kernel_sizes if kernel_sizes else [3] * len(channels)
        self.strides = strides if strides else [1] * len(channels)
        self.conv_layers = nn.ModuleDict()

        for i_layer, (in_channels, out_channels, kernel_size, stride) in enumerate(
            zip(
                [input_shape[0]] + channels[:-1],
                channels,
                self.kernel_sizes,
                self.strides,
            )
        ):
            self.conv_layers[f"conv_{i_layer}"] = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding="same",
            )
            self.conv_layers[f"relu_{i_layer}"] = nn.ReLU()
            self.conv_layers[f"pool_{i_layer}"] = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.mlp = MLP((self.channels[-1],), self.output_dim, self.dense_dims)

    def forward(self, x):
        for name, layer in self.conv_layers.items():
            x = layer(x)
            self.store(f"conv_post_{name}", x)
        x = self.global_pool(x)
        self.store("post_global_pool", x)
        x = self.call_submodule("mlp", x)
        return x

    @property
    def default_names(self) -> list[str]:
        return [
            f"conv_post_{name}"
            for name in self.conv_layers.keys()
            if name.startswith("conv")
        ] + ["mlp_" + name for name in self.mlp.default_names]


########################################################################################
# Modified from https://github.com/kuangliu/pytorch-cifar/
# License of original:
#
# MIT License
#
# Copyright (c) 2017 liukuang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
########################################################################################


class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )

        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    self.expansion * out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(HookedModel):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # TODO: should capture more activations, including from inside blocks
        out = self.conv1(x)
        out = self.layer1(out)
        self.store("res1", out)
        out = self.layer2(out)
        self.store("res2", out)
        out = self.layer3(out)
        self.store("res3", out)
        out = self.layer4(out)
        self.store("res4", out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
