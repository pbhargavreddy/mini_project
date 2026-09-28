"""
Similar U-net, variational autoencoder
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class MiniUNetDehaze(nn.Module):

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        # Encoder
        self.inc = DoubleConv(in_channels, 32)

        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(32, 64)
        )

        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128)
        )

        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256)
        )

        # Decoder
        self.up1 = nn.ConvTranspose2d(
            256, 128, kernel_size=2, stride=2
        )

        self.conv_up1 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2
        )

        self.conv_up2 = DoubleConv(128, 64)

        self.up3 = nn.ConvTranspose2d(
            64, 32, kernel_size=2, stride=2
        )

        self.conv_up3 = DoubleConv(64, 32)

        # Output
        self.outc = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):

        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Decoder
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)

        return self.outc(x)