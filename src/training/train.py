import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.encoding.tabular_to_image import CorrelationAwareEncoder
from src.models.hybrid_cnn_vit import HybridCNNViT

class Config:
    DATA_PATH = "data/raw/retail_dataset.csv"  
    IMAGE_SIZE = 32

def main():
    config = Config()
    
    # Load data (adjust column names if needed)
    df = pd.read_csv(config.DATA_PATH)
    feature_cols = ['Price', 'Stock_Quantity', 'Warranty_Period', 'Rating']
    target_col = 'Category'   # Change if your target column name is different
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Correlation-Aware Encoder (Core Contribution)
    encoder = CorrelationAwareEncoder(image_size=config.IMAGE_SIZE)
    encoder.fit(X_train, feature_names=feature_cols)
    encoder.visualize_placement()
    
    X_train_img = encoder.transform(X_train)
    X_test_img = encoder.transform(X_test)
    
    # To Tensor
    train_ds = TensorDataset(torch.tensor(X_train_img), torch.tensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    
    model = HybridCNNViT(num_classes=len(le.classes_))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    print("Starting training...")
    for epoch in range(5):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} completed")
    
    print("✅ Training finished!")
    torch.save(model.state_dict(), "results/model_final.pth")

if __name__ == "__main__":
    main()
