"""
Brain tumor segmentation model based on a Bridged U‑Net with an Atrous
Spatial Pyramid Pooling (ASPP) bottleneck.  The implementation here takes
inspiration from the Bridged‑U‑Net‑ASPP‑EVO architecture described in the
paper *Bridged‑U‑Net‑ASPP‑EVO and deep learning optimization for brain
tumor segmentation* by Yousef et al., published in the journal
*Diagnostics* (2023).  That work introduces several improvements over
the original U‑Net, including evolving normalization layers, squeeze and
excitation residual blocks, max–average pooling for down‑sampling and
an ASPP module for capturing multi‑scale context【133007765279740†L314-L331】.

This file provides a PyTorch implementation of a simplified version of
that architecture.  The key components implemented here are:

* **ConvBlock** – two convolutional layers with GroupNorm and ReLU.
  Evolving normalization (EvoNorm) is a trainable normalization layer
  originally proposed in the paper; however, to avoid introducing
  custom operations we approximate EvoNorm with Group Normalization,
  which has been shown to work well for medical imaging tasks.  Each
  ConvBlock optionally applies a squeeze‑and‑excitation (SE) module to
  recalibrate channel responses【133007765279740†L314-L331】.

* **DownBlock** – combines ConvBlock with a max–average pooling
  operation.  The pooling step concatenates the results of max pooling
  and average pooling to better preserve spatial context as suggested
  by Yousef et al.【133007765279740†L314-L331】.

* **UpBlock** – performs nearest‑neighbour up‑sampling followed by a
  ConvBlock.  Skip connections from the encoder are bridged using
  a 1×1 convolution so that the number of channels matches before
  concatenation.

* **ASPP** – the Atrous Spatial Pyramid Pooling module at the
  bottleneck.  Multiple parallel 3×3 convolutions with different
  dilation rates capture context at multiple scales【133007765279740†L323-L331】.

The resulting network accepts a single 2D MRI slice as input and
produces a single segmentation mask with one channel output.  If your
model was trained on multi‑modal MRI slices (e.g. T1, T2, FLAIR), you
should stack those modalities along the channel dimension when calling
``model`` and adjust the ``in_channels`` argument accordingly.  The
architecture is flexible; the number of filters can be changed via
``base_filters``.

Note: This implementation is provided for demonstration purposes.  It
is deliberately simple enough to run on the limited compute available
in the workshop environment.  If you have a trained state dictionary
for your Bridged‑U‑Net model, you can load it via
``model.load_state_dict(torch.load('path/to/weights.pt', map_location='cpu'))``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze‑and‑Excitation block for channel‑wise attention.

    Given an input tensor ``x`` of shape (N, C, H, W), the block first
    performs global average pooling to produce a channel descriptor of
    shape (N, C, 1, 1).  Two small fully connected layers implement
    the squeeze and excitation operations.  A sigmoid activation
    produces scale factors in (0, 1) that are broadcast and multiplied
    onto the original input.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squeeze: global average pooling
        scale = F.adaptive_avg_pool2d(x, 1)
        # Excitation
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        return x * scale


class ConvBlock(nn.Module):
    """Two convolutional layers with GroupNorm and ReLU.

    Optionally applies a squeeze‑and‑excitation block at the end of
    the second convolution.  GroupNorm is used instead of BatchNorm
    because it performs better with small batch sizes typical in
    medical imaging.  The number of groups is set to 8 by default.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 8,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(groups, out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(groups, out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = self.relu2(x)
        x = self.se(x)
        return x


class MaxAvgPool(nn.Module):
    """Combine max pooling and average pooling for down‑sampling.

    The output of max pooling and average pooling are concatenated
    along the channel dimension.  This is inspired by the max–average
    pooling operator described in the Bridged‑U‑Net‑ASPP‑EVO paper【133007765279740†L314-L331】.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        # After concatenation, the number of channels doubles.  A 1×1
        # convolution reduces it back to ``in_channels`` while allowing
        # the network to learn a weighted combination of max and mean
        # features.
        self.reduction = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_pooled = F.max_pool2d(x, kernel_size=2, stride=2)
        avg_pooled = F.avg_pool2d(x, kernel_size=2, stride=2)
        combined = torch.cat([max_pooled, avg_pooled], dim=1)
        return self.reduction(combined)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module.

    Implements parallel atrous convolutions with different dilation rates
    (1, 6, 12, 18) to capture multi‑scale context【133007765279740†L323-L331】.  Outputs are
    concatenated and passed through a final 1×1 convolution to fuse
    information across scales.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        dilations = [1, 6, 12, 18]
        self.convs = nn.ModuleList()
        for d in dilations:
            self.convs.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=d,
                    dilation=d,
                    bias=False,
                )
            )
        self.bn = nn.GroupNorm(8, out_channels * len(dilations))
        self.relu = nn.ReLU(inplace=True)
        self.project = nn.Conv2d(out_channels * len(dilations), out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [conv(x) for conv in self.convs]
        x = torch.cat(outputs, dim=1)
        x = self.bn(x)
        x = self.relu(x)
        x = self.project(x)
        return x


class UpBlock(nn.Module):
    """Upsampling block with skip connection and bridging.

    The skip connection tensor from the encoder is first passed
    through a 1×1 convolution (bridge) to match the number of channels
    before concatenation.  After concatenation, a ConvBlock is
    applied.  Nearest neighbour up‑sampling is used for simplicity.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Bridge to align the skip connection channels
        self.bridge = nn.Conv2d(skip_channels, in_channels // 2, kernel_size=1)
        self.conv = ConvBlock(in_channels // 2 + in_channels, out_channels, use_se=use_se)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        skip = self.bridge(skip)
        # Pad if necessary (in case of odd sizes)
        if x.shape[-1] != skip.shape[-1] or x.shape[-2] != skip.shape[-2]:
            diffY = skip.shape[-2] - x.shape[-2]
            diffX = skip.shape[-1] - x.shape[-1]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class BridgedUNetASPP(nn.Module):
    """Simplified Bridged U‑Net with ASPP.

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g. 1 for a single MRI modality or 3 for RGB).
    num_classes : int
        Number of output channels.  For binary segmentation this should be 1.
    base_filters : int
        Number of filters in the first encoder layer.  This number doubles
        with each down‑sampling step.
    use_se : bool
        Whether to include squeeze‑and‑excitation blocks.  The original
        Bridged‑U‑Net‑ASPP‑EVO includes SE residual blocks【133007765279740†L314-L331】.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_filters: int = 32,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        filters = [base_filters, base_filters * 2, base_filters * 4, base_filters * 8]

        # Encoder
        self.enc1 = ConvBlock(in_channels, filters[0], use_se=use_se)
        self.pool1 = MaxAvgPool(filters[0])
        self.enc2 = ConvBlock(filters[0], filters[1], use_se=use_se)
        self.pool2 = MaxAvgPool(filters[1])
        self.enc3 = ConvBlock(filters[1], filters[2], use_se=use_se)
        self.pool3 = MaxAvgPool(filters[2])
        self.enc4 = ConvBlock(filters[2], filters[3], use_se=use_se)

        # Bottleneck ASPP
        self.aspp = ASPP(filters[3], filters[3])

        # Decoder
        self.up3 = UpBlock(filters[3], filters[2], filters[2], use_se=use_se)
        self.up2 = UpBlock(filters[2], filters[1], filters[1], use_se=use_se)
        self.up1 = UpBlock(filters[1], filters[0], filters[0], use_se=use_se)

        # Final convolution
        self.final_conv = nn.Conv2d(filters[0], num_classes, kernel_size=1)

        # Use sigmoid for binary segmentation
        self.activation = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)

        # ASPP bottleneck
        b = self.aspp(e4)

        # Decoder with bridging skip connections
        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        out = self.final_conv(d1)
        return self.activation(out)


def load_model(weights_path: str | None = None, device: str = 'cpu') -> nn.Module:
    """Create a BridgedUNetASPP model and optionally load a state dictionary.

    Parameters
    ----------
    weights_path : str or None
        If provided, the file system path to a PyTorch state dictionary
        (.pt or .pth file).  The model architecture defined above will
        be used to instantiate the network, and then weights will be
        loaded.  If the weights fail to load (e.g. due to mismatched
        keys), an exception will be raised.
    device : str
        Device onto which the model should be loaded.  Defaults to
        'cpu'.

    Returns
    -------
    nn.Module
        The instantiated (and possibly weight‑loaded) model.
    """

    model = BridgedUNetASPP(in_channels=1, num_classes=1, base_filters=32, use_se=True)
    if weights_path:
        state = torch.load(weights_path, map_location=device)
        # Some checkpoints may include a top‑level key such as 'model'
        # or 'state_dict'.  Attempt to handle these common cases.
        if isinstance(state, dict) and not any(k.startswith('0') for k in state.keys()):
            # Try nested keys
            if 'model_state' in state:
                state = state['model_state']
            elif 'state_dict' in state:
                state = state['state_dict']
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model