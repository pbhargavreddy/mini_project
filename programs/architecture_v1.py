"""
Spacially adaptive 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Lightweight CNN Stem
# ============================================================

class Stem(nn.Module):

    def __init__(self, in_channels=3, features=32):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# 2. Spatially Adaptive Multi-Scale Block
# ============================================================

class SpatialAdaptiveMultiScaleBlock(nn.Module):

    def __init__(self, channels=32):
        super().__init__()

        # Local scale
        self.branch_3x3 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels
        )

        # Medium scale
        self.branch_5x5 = nn.Conv2d(
            channels,
            channels,
            kernel_size=5,
            padding=2,
            groups=channels
        )

        # Large receptive field
        self.branch_dilated = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=channels
        )

        # Spatial scale gating
        self.gate = nn.Sequential(
            nn.Conv2d(
                channels * 3,
                channels,
                kernel_size=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                3,
                kernel_size=1
            )
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        f1 = self.branch_3x3(x)
        f2 = self.branch_5x5(x)
        f3 = self.branch_dilated(x)

        combined = torch.cat(
            [f1, f2, f3],
            dim=1
        )

        weights = self.gate(combined)
8
        weights = self.softmax(weights)

        w1 = weights[:, 0:1, :, :]
        w2 = weights[:, 1:2, :, :]
        w3 = weights[:, 2:3, :, :]

        out = (
            w1 * f1 +
            w2 * f2 +
            w3 * f3
        )

        return out


# ============================================================
# 3. Channel Attention
# ============================================================

class ChannelAttention(nn.Module):

    def __init__(self, channels=32, reduction=8):
        super().__init__()

        hidden = channels // reduction

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden,
                kernel_size=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden,
                channels,
                kernel_size=1
            ),
            nn.Sigmoid()
        )

    def forward(self, x):

        pooled = self.pool(x)

        weights = self.fc(pooled)

        return x * weights


# ============================================================
# 4. Spatial Attention
# ============================================================

class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_map = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        max_map, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        combined = torch.cat(
            [avg_map, max_map],
            dim=1
        )

        attention = self.sigmoid(
            self.conv(combined)
        )

        return x * attention


# ============================================================
# 5. Residual Refinement
# ============================================================

class ResidualRefinement(nn.Module):

    def __init__(self, channels=32):
        super().__init__()

        self.refine = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x):

        residual = self.refine(x)

        return x + residual


# ============================================================
# 6. Combined Feature Block
# ============================================================

class FeatureBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.multi_scale = SpatialAdaptiveMultiScaleBlock(
            channels
        )

        self.channel_attention = ChannelAttention(
            channels
        )

        self.spatial_attention = SpatialAttention()

        self.refinement = ResidualRefinement(
            channels
        )

    def forward(self, x):

        x = self.multi_scale(x)

        x = self.channel_attention(x)

        x = self.spatial_attention(x)

        x = self.refinement(x)

        return x


# ============================================================
# 7. Downsampling Block
# ============================================================

class DownsampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.block(x)


# ============================================================
# 8. Upsampling Block
# ============================================================

class UpsampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

        return self.conv(x)


# ============================================================
# 9. Lightweight U-Net
# ============================================================

class LightweightUNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder
        self.down1 = DownsampleBlock(32, 64)

        self.encoder1 = FeatureBlock(64)

        self.down2 = DownsampleBlock(64, 96)

        self.encoder2 = FeatureBlock(96)

        # Bottleneck
        self.bottleneck = FeatureBlock(96)

        # Decoder
        self.up1 = UpsampleBlock(96, 64)

        self.decoder1 = nn.Sequential(

            nn.Conv2d(
                64 + 64,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True)
        )

        self.up2 = UpsampleBlock(64, 32)

        self.decoder2 = nn.Sequential(

            nn.Conv2d(
                32 + 32,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        # 256x256
        skip1 = x

        # 256 -> 128
        x = self.down1(x)

        skip2 = self.encoder1(x)

        # 128 -> 64
        x = self.down2(skip2)

        x = self.encoder2(x)

        # Bottleneck
        x = self.bottleneck(x)

        # 64 -> 128
        x = self.up1(x)

        x = torch.cat(
            [x, skip2],
            dim=1
        )

        x = self.decoder1(x)

        # 128 -> 256
        x = self.up2(x)

        x = torch.cat(
            [x, skip1],
            dim=1
        )

        x = self.decoder2(x)

        return x


# ============================================================
# 10. Reconstruction Head
# ============================================================

class ReconstructionHead(nn.Module):

    def __init__(self, in_channels=32):
        super().__init__()

        self.head = nn.Conv2d(
            in_channels,
            3,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        return self.head(x)


# ============================================================
# 11. Complete Dehazing Network
# ============================================================
class DehazingNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.stem = Stem(
            in_channels=3,
            features=32
        )

        self.feature_block = FeatureBlock(32)

        self.unet = LightweightUNet()

        self.reconstruction = ReconstructionHead(32)

    def forward(self, x):

        # Keep original hazy image
        identity = x

        x = self.stem(x)

        x = self.feature_block(x)

        x = self.unet(x)

        # Predict residual correction
        residual = self.reconstruction(x)

        # Hazy + learned correction
        output = identity + residual

        # Keep output in valid image range
        output = torch.clamp(output, 0, 1)

        return output