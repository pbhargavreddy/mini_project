"""
Adaptive Routing
                     HAZY RGB IMAGE
                           │
                           ▼
                ┌────────────────────┐
                │  LIGHTWEIGHT CNN   │
                │       STEM         │
                └─────────┬──────────┘
                          │
                          ▼
                  ADAPTIVE ROUTER
                          │
                   ┌──────┴──────┐
                   │             │
                 CLEAR          HAZY
                   │             │
                   ▼             ▼
                BYPASS      MULTI-SCALE
                             DEHAZING
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
                  3×3 DW     Dilated     5×5 DW
                             3×3 DW
                    └───────────┼───────────┘
                                │
                                ▼
                       SPATIAL SCALE GATING
                                │
                         w₁(x,y), w₂(x,y), w₃(x,y)
                                │
                                ▼
                         WEIGHTED FUSION
                                │
                                ▼
                    LIGHTWEIGHT ATTENTION
                                │
                                ▼
                       RESIDUAL REFINEMENT
                                │
                                ▼
                       LIGHTWEIGHT DECODER
                                │
                                ▼
                           CLEAR IMAGE


"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. BASIC BUILDING BLOCKS
# ============================================================

class ConvBNAct(nn.Module):
    """
    Standard lightweight convolution block:
    Conv -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Convolution + Pointwise Convolution

    Much cheaper than a standard convolution.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1
    ):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False
        )

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)

        return x


# ============================================================
# 2. LIGHTWEIGHT CNN STEM
# ============================================================

class LightweightStem(nn.Module):
    """
    Extracts low-level features from the hazy RGB image.
    """

    def __init__(self, in_channels=3, feature_channels=64):
        super().__init__()

        self.stem = nn.Sequential(
            ConvBNAct(
                in_channels,
                32,
                kernel_size=3,
                padding=1
            ),

            DepthwiseSeparableConv(
                32,
                feature_channels,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x):
        return self.stem(x)


# ============================================================
# 3. ADAPTIVE ROUTER
# ============================================================

class AdaptiveRouter(nn.Module):
    """
    Predicts a spatial haze probability.

    Output:
        routing_map ∈ [0, 1]

    0 -> clear
    1 -> hazy
    """

    def __init__(self, channels):
        super().__init__()

        self.router = nn.Sequential(
            nn.Conv2d(
                channels,
                channels // 2,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels // 2,
                1,
                kernel_size=1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):
        return self.router(x)


# ============================================================
# 4. MULTI-SCALE DEHAZING BLOCK
# ============================================================

class MultiScaleDehazing(nn.Module):
    """
    Three parallel spatial scales:

        Branch 1 -> 3x3 Depthwise
        Branch 2 -> Dilated 3x3 Depthwise
        Branch 3 -> 5x5 Depthwise
    """

    def __init__(self, channels):
        super().__init__()

        # ----------------------------------------------------
        # Scale 1: Fine details
        # ----------------------------------------------------
        self.branch_3x3 = DepthwiseSeparableConv(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # Scale 2: Larger receptive field
        # ----------------------------------------------------
        self.branch_dilated = DepthwiseSeparableConv(
            channels,
            channels,
            kernel_size=3,
            padding=2,
            dilation=2
        )

        # ----------------------------------------------------
        # Scale 3: Larger local context
        # ----------------------------------------------------
        self.branch_5x5 = DepthwiseSeparableConv(
            channels,
            channels,
            kernel_size=5,
            padding=2
        )

    def forward(self, x):

        f1 = self.branch_3x3(x)

        f2 = self.branch_dilated(x)

        f3 = self.branch_5x5(x)

        return f1, f2, f3


# ============================================================
# 5. SPATIAL SCALE GATING
# ============================================================

class SpatialScaleGating(nn.Module):
    """
    Generates three spatial weights:

        w1(x,y)
        w2(x,y)
        w3(x,y)

    Softmax guarantees:

        w1 + w2 + w3 = 1
    """

    def __init__(self, channels):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Conv2d(
                channels * 3,
                channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                3,
                kernel_size=1
            )
        )

    def forward(self, f1, f2, f3):

        # Concatenate multi-scale features
        combined = torch.cat(
            [f1, f2, f3],
            dim=1
        )

        # Generate 3 spatial scale maps
        weights = self.gate(combined)

        # Normalize scale weights
        weights = F.softmax(
            weights,
            dim=1
        )

        w1 = weights[:, 0:1, :, :]
        w2 = weights[:, 1:2, :, :]
        w3 = weights[:, 2:3, :, :]

        # Weighted fusion
        fused = (
            w1 * f1 +
            w2 * f2 +
            w3 * f3
        )

        return fused, weights


# ============================================================
# 6. LIGHTWEIGHT ATTENTION
# ============================================================

class LightweightAttention(nn.Module):
    """
    Lightweight channel attention.

    Uses global average pooling to determine
    which feature channels are important.
    """

    def __init__(self, channels, reduction=8):
        super().__init__()

        hidden = max(channels // reduction, 4)

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

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

        weights = self.attention(x)

        return x * weights


# ============================================================
# 7. RESIDUAL REFINEMENT
# ============================================================

class ResidualRefinement(nn.Module):
    """
    Learns restoration corrections instead of
    reconstructing everything from scratch.

        output = input + residual
    """ 

    def __init__(self, channels):
        super().__init__()

        self.refinement = nn.Sequential(

            DepthwiseSeparableConv(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),

            DepthwiseSeparableConv(
                channels,
                channels,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x):

        residual = self.refinement(x)

        return x + residual


# ============================================================
# 8. LIGHTWEIGHT DECODER
# ============================================================

class LightweightDecoder(nn.Module):
    """
    Converts feature representation back to RGB.
    """

    def __init__(self, channels, out_channels=3):
        super().__init__()

        self.decoder = nn.Sequential(

            DepthwiseSeparableConv(
                channels,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.Conv2d(
                32,
                out_channels,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x):

        return self.decoder(x)


# ============================================================
# 9. COMPLETE DEHAZING NETWORK
# ============================================================

class LightweightSpatialAdaptiveDehazer(nn.Module):
    """
    Lightweight Spatially Adaptive Multi-Scale
    Image Dehazing Network

    Pipeline:

        Hazy RGB
            ↓
        Lightweight Stem
            ↓
        Adaptive Router
            ↓
        ┌───────────────┐
        │               │
      Clear           Hazy
        │               │
      Bypass      Multi-Scale
                      ↓
               Spatial Gating
                      ↓
                Weighted Fusion
                      ↓
              Lightweight Attention
                      ↓
              Residual Refinement
                      ↓
              Lightweight Decoder
                      ↓
                  Clear RGB
    """

    def __init__(
        self,
        in_channels=3,
        feature_channels=64
    ):
        super().__init__()

        # ----------------------------------------------------
        # Stem
        # ----------------------------------------------------
        self.stem = LightweightStem(
            in_channels=in_channels,
            feature_channels=feature_channels
        )

        # ----------------------------------------------------
        # Adaptive Router
        # ----------------------------------------------------
        self.router = AdaptiveRouter(
            feature_channels
        )

        # ----------------------------------------------------
        # Multi-scale processing
        # ----------------------------------------------------
        self.multi_scale = MultiScaleDehazing(
            feature_channels
        )

        # ----------------------------------------------------
        # Spatial scale gating
        # ----------------------------------------------------
        self.scale_gate = SpatialScaleGating(
            feature_channels
        )

        # ----------------------------------------------------
        # Lightweight attention
        # ----------------------------------------------------
        self.attention = LightweightAttention(
            feature_channels
        )

        # ----------------------------------------------------
        # Residual refinement
        # ----------------------------------------------------
        self.refinement = ResidualRefinement(
            feature_channels
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------
        self.decoder = LightweightDecoder(
            feature_channels,
            out_channels=3
        )

    def forward(self, x, return_aux=False):

        # ====================================================
        # STEP 1: FEATURE EXTRACTION
        # ====================================================

        features = self.stem(x)

        # ====================================================
        # STEP 2: ADAPTIVE ROUTING
        # ====================================================

        haze_map = self.router(features)

        # ====================================================
        # STEP 3: MULTI-SCALE DEHAZING
        # ====================================================

        f1, f2, f3 = self.multi_scale(features)

        # ====================================================
        # STEP 4: SPATIAL SCALE GATING
        # ====================================================

        dehazed_features, scale_weights = self.scale_gate(
            f1,
            f2,
            f3
        )

        # ====================================================
        # STEP 5: ROUTING
        # ====================================================

        # haze_map:
        #
        # 0 -> clear
        # 1 -> hazy
        #
        # Clear regions retain original features.
        # Hazy regions use dehazed features.

        routed_features = (
            (1.0 - haze_map) * features
            +
            haze_map * dehazed_features
        )

        # ====================================================
        # STEP 6: LIGHTWEIGHT ATTENTION
        # ====================================================

        attended_features = self.attention(
            routed_features
        )

        # ====================================================
        # STEP 7: RESIDUAL REFINEMENT
        # ====================================================

        refined_features = self.refinement(
            attended_features
        )

        # ====================================================
        # STEP 8: RGB RECONSTRUCTION
        # ====================================================

        residual_image = self.decoder(
            refined_features
        )

        # ====================================================
        # STEP 9: GLOBAL RESIDUAL LEARNING
        # ====================================================

        output = x + residual_image

        # Keep output in valid RGB range
        output = torch.clamp(
            output,
            0.0,
            1.0
        )

        # ====================================================
        # OPTIONAL DEBUG INFORMATION
        # ====================================================

        if return_aux:

            return {
                "output": output,
                "haze_map": haze_map,
                "scale_weights": scale_weights,
                "stem_features": features,
                "multi_scale_features": (
                    f1,
                    f2,
                    f3
                )
            }

        return output


# ============================================================
# 10. MODEL INFORMATION
# ============================================================

def count_parameters(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


# ============================================================
# 11. TEST THE NETWORK
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = LightweightSpatialAdaptiveDehazer(
        in_channels=3,
        feature_channels=64
    ).to(device)

    # --------------------------------------------------------
    # Count parameters
    # --------------------------------------------------------

    total_params, trainable_params = count_parameters(
        model
    )

    print("\nModel Parameters")
    print("----------------------------")
    print(
        f"Total parameters: "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    # --------------------------------------------------------
    # Dummy input
    # --------------------------------------------------------

    x = torch.rand(
        1,
        3,
        256,
        256
    ).to(device)

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        result = model(
            x,
            return_aux=True
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output = result["output"]
    haze_map = result["haze_map"]
    scale_weights = result["scale_weights"]

    print("\nTensor Shapes")
    print("----------------------------")

    print("Input:",
          x.shape)

    print("Output:",
          output.shape)

    print("Haze map:",
          haze_map.shape)

    print("Scale weights:",
          scale_weights.shape)

    print("\nValue Ranges")
    print("----------------------------")

    print(
        "Input range:",
        x.min().item(),
        x.max().item()
    )

    print(
        "Output range:",
        output.min().item(),
        output.max().item()
    )

    print(
        "Haze map range:",
        haze_map.min().item(),
        haze_map.max().item()
    )

    print("\nAverage Scale Weights")
    print("----------------------------")

    avg_weights = (
        scale_weights
        .mean(dim=(0, 2, 3))
    )

    print(
        "3x3 DW:",
        avg_weights[0].item()
    )

    print(
        "Dilated 3x3 DW:",
        avg_weights[1].item()
    )

    print(
        "5x5 DW:",
        avg_weights[2].item()
    )

    print("\nModel test completed.")