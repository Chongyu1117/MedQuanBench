import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(Conv3D => BatchNorm3d => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with strided convolution followed by double conv"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down_conv = nn.Sequential(
            # Strided convolution to reduce spatial dimensions
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            # Additional convolution layer
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.down_conv(x)

class Up(nn.Module):
    """Upscaling using ConvTranspose3d with adjusted output_padding followed by double conv"""

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        # Adjust output_padding to ensure the output size matches the encoder feature map size
        self.up = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 is the input from the decoder path (after upsampling)
        # x2 is the corresponding feature map from the encoder path (for concatenation)
        x1 = self.up(x1)
        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Final 1x1x1 convolution layer to get the desired output channels"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    """3D U-Net architecture compatible with TensorRT and ONNX export"""

    def __init__(self, n_channels, n_classes, base_features=16):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        features = [
            base_features,       # 16
            base_features * 2,   # 32
            base_features * 4,   # 64
            base_features * 8,   # 128
            base_features * 16   # 256
        ]

        self.inc = DoubleConv(n_channels, features[0])
        self.down1 = Down(features[0], features[1])  # 16 -> 32
        self.down2 = Down(features[1], features[2])  # 32 -> 64
        self.down3 = Down(features[2], features[3])  # 64 -> 128
        self.down4 = Down(features[3], features[4])  # 128 -> 256

        self.up1 = Up(features[4], features[3])      # 256 -> 128
        self.up2 = Up(features[3], features[2])      # 128 -> 64
        self.up3 = Up(features[2], features[1])      # 64 -> 32
        self.up4 = Up(features[1], features[0])      # 32 -> 16

        self.outc = OutConv(features[0], n_classes)

    def forward(self, x):
        x1 = self.inc(x)     # Level 1
        x2 = self.down1(x1)  # Level 2
        x3 = self.down2(x2)  # Level 3
        x4 = self.down3(x3)  # Level 4
        x5 = self.down4(x4)  # Bottleneck

        x = self.up1(x5, x4)  # Level 4
        x = self.up2(x, x3)   # Level 3
        x = self.up3(x, x2)   # Level 2
        x = self.up4(x, x1)   # Level 1

        logits = self.outc(x)
        return logits
