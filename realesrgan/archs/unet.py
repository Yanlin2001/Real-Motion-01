"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
from torch.nn import functional as F
from basicsr.utils.registry import ARCH_REGISTRY # +

class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, num_in_ch: int, num_out_ch: int, drop_prob: float):
        """
        Args:
            num_in_ch: Number of channels in the input.
            num_out_ch: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(num_out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(num_out_ch, num_out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(num_out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, num_in_ch, H, W)`.

        Returns:
            Output tensor of shape `(N, num_out_ch, H, W)`.
        """
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, num_in_ch: int, num_out_ch: int):
        """
        Args:
            num_in_ch: Number of channels in the input.
            num_out_ch: Number of channels in the output.
        """
        super().__init__()

        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                num_in_ch, num_out_ch, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(num_out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, num_in_ch, H, W)`.

        Returns:
            Output tensor of shape `(N, num_out_ch, H*2, W*2)`.
        """
        return self.layers(image)


@ARCH_REGISTRY.register() # +
class Unet(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        num_in_ch: int,
        num_out_ch: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            num_in_ch: Number of channels in the input to the U-Net model.
            num_out_ch: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super(Unet, self).__init__()

        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(num_in_ch, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.num_out_ch, kernel_size=1, stride=1),
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, num_in_ch, H, W)`.

        Returns:
            Output tensor of shape `(N, num_out_ch, H, W)`.
        """
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output


