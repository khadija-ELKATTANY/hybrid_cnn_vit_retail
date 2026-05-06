import sys
sys.path.append('src')

import torch
from encoding.tabular_to_image import TabularToImageEncoder
from models.hybrid_cnn_vit import HybridCNNViT

print("✅ Basic imports successful!")

# Test encoder
encoder = TabularToImageEncoder(grid_size=32)
print("✅ Encoder initialized")

# Test model
model = HybridCNNViT(num_classes=8)
print("✅ Model initialized")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Dummy forward pass
dummy_image = torch.randn(1, 3, 32, 32)
output = model(dummy_image)
print(f"✅ Forward pass successful! Output shape: {output.shape}")