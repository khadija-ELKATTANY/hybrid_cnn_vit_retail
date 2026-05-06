import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import roc_curve, auc

from src.encoding.tabular_to_image import CorrelationAwareEncoder
from src.models.hybrid_cnn_vit import HybridCNNViT


DATA_PATH = "data/generated/products_structured.csv"
MODEL_PATH = "results/model_final.pth"
OUT_PATH = "results/figures/roc_curves.png"

feature_cols = ["Price", "Stock Quantity", "Warranty Period", "Product Ratings"]
target_col = "Product Category"

df = pd.read_csv(DATA_PATH)

X = df[feature_cols].values
y_raw = df[target_col].values

le = LabelEncoder()
y = le.fit_transform(y_raw)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

encoder = CorrelationAwareEncoder(image_size=32)
encoder.fit(X_train, feature_names=feature_cols)

X_test_img = encoder.transform(X_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridCNNViT(num_classes=len(le.classes_)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

with torch.no_grad():
    xb = torch.tensor(X_test_img).float().to(device)
    logits = model(xb)
    probs = torch.softmax(logits, dim=1).cpu().numpy()

y_bin = label_binarize(y_test, classes=np.arange(len(le.classes_)))

plt.figure(figsize=(10, 8))

for i, class_name in enumerate(le.classes_):
    fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.500)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Figure 4.3: Per-class ROC curves on the test set (one-vs-rest)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()

Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, dpi=300)
plt.close()

print(f"Saved: {OUT_PATH}")
