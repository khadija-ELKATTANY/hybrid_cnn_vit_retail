# src/models/hybrid_cnn_vit.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CNNStem(nn.Module):
    """Residual CNN Backbone"""
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(base_channels, base_channels, blocks=2, stride=1)
        self.layer2 = self._make_layer(base_channels, base_channels*2, blocks=2, stride=2)
        self.layer3 = self._make_layer(base_channels*2, base_channels*4, blocks=2, stride=2)

    def _make_layer(self, in_ch: int, out_ch: int, blocks: int, stride: int):
        layers = [ResidualBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class PatchEmbedding(nn.Module):
    """Convolutional Patch Embedding for ViT"""
    def __init__(self, in_channels: int = 256, patch_size: int = 2, embed_dim: int = 384):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + (16*16), embed_dim))  # assuming after CNN we get ~16x16 feature map

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x)                    # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)    # (B, N, embed_dim)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int = 384, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Pre-LN
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class HybridCNNViT(nn.Module):
    def __init__(self, num_classes: int = 8, embed_dim: int = 384, num_heads: int = 8, num_layers: int = 4):
        super().__init__()
        
        self.cnn = CNNStem(in_channels=3, base_channels=64)
        self.patch_embed = PatchEmbedding(in_channels=256, patch_size=2, embed_dim=embed_dim)
        
        self.transformer = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        # CNN Stem
        x = self.cnn(x)                    # (B, 256, H', W')
        
        # Patch Embedding + CLS token
        x = self.patch_embed(x)            # (B, N+1, embed_dim)
        
        # Transformer Encoder
        for block in self.transformer:
            x = block(x)
        
        x = self.norm(x)
        cls_token = x[:, 0]                # Use CLS token
        return self.head(cls_token)