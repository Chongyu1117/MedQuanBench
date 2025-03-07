# transunet.py

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=scale_factor, stride=scale_factor)

    def forward(self, x):
        return self.up(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # src shape: (N, B, E)
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.activation(self.linear1(src)))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransUNet3D(nn.Module):
    def __init__(self, img_size=(96, 96, 96), in_channels=1, num_classes=14, base_channels=32,
                 embed_dim=256, mlp_dim=1024, num_heads=4, num_layers=4, dropout=0.1):
        super(TransUNet3D, self).__init__()
        self.img_size = img_size  # (D, H, W)
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Encoder with strided convolutions instead of pooling
        self.conv1 = ConvBlock(in_channels, base_channels, stride=1)
        self.conv2 = ConvBlock(base_channels, base_channels * 2, stride=2)
        self.conv3 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)
        self.conv4 = ConvBlock(base_channels * 4, base_channels * 8, stride=2)

        # Bottleneck
        self.conv5 = ConvBlock(base_channels * 8, base_channels * 16, stride=2)

        # Transformer Encoder
        self.flatten = nn.Flatten(2)
        num_patches = (img_size[0] // 16) * (img_size[1] // 16) * (img_size[2] // 16)
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        self.embed_conv = nn.Conv3d(base_channels * 16, embed_dim, kernel_size=1)

        # Decoder
        self.upconv4 = UpConv(embed_dim, base_channels * 8, scale_factor=2)
        self.dec4 = ConvBlock(base_channels * 16, base_channels * 8)

        self.upconv3 = UpConv(base_channels * 8, base_channels * 4, scale_factor=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)

        self.upconv2 = UpConv(base_channels * 4, base_channels * 2, scale_factor=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)

        self.upconv1 = UpConv(base_channels * 2, base_channels, scale_factor=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        # Final classification layer
        self.final_conv = nn.Conv3d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)   # No downsampling, shape remains the same
        c2 = self.conv2(c1)  # Downsampled by factor of 2
        c3 = self.conv3(c2)  # Downsampled by factor of 4
        c4 = self.conv4(c3)  # Downsampled by factor of 8
        c5 = self.conv5(c4)  # Downsampled by factor of 16

        # Transformer Encoder
        x_embed = self.embed_conv(c5)
        B, C, D, H, W = x_embed.shape
        x_flat = self.flatten(x_embed)  # Shape: (B, C, N)
        x_flat = x_flat.permute(2, 0, 1)  # Shape: (N, B, C)
        x_flat = x_flat + self.position_embeddings.permute(1, 0, 2)  # Add position embeddings

        for transformer in self.transformer_blocks:
            x_flat = transformer(x_flat)

        x_out = x_flat.permute(1, 2, 0)  # Shape: (B, C, N)
        x_out = x_out.view(B, C, D, H, W)  # Reshape back to (B, C, D, H, W)

        # Decoder
        d4 = self.upconv4(x_out)  # Upsample
        d4 = torch.cat((d4, c4), dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, c3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, c2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, c1), dim=1)
        d1 = self.dec1(d1)

        # Final classification layer
        out = self.final_conv(d1)
        return out
