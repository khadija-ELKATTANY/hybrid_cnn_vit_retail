from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.encoding.tabular_to_image import CorrelationAwareEncoder
from src.models.hybrid_cnn_vit import HybridCNNViT


DATA_PATH = "data/generated/products_structured.csv"
MODEL_PATH = "results/model_final.pth"
OUT_PATH = "results/figures/latency_distribution.png"

feature_cols = [
    "Price",
    "Stock Quantity",
    "Warranty Period",
    "Product Ratings",
]

target_col = "Product Category"

# Load data
df = pd.read_csv(DATA_PATH)

X = df[feature_cols].values
y_raw = df[target_col].values

le = LabelEncoder()
y = le.fit_transform(y_raw)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

encoder = CorrelationAwareEncoder(image_size=32)
encoder.fit(X_train, feature_names=feature_cols)

X_test_img = encoder.transform(X_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HybridCNNViT(num_classes=len(le.classes_)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

sample = torch.tensor(X_test_img[:1]).float().to(device)

# Warm-up
with torch.no_grad():
    for _ in range(20):
        _ = model(sample)

latencies = []

with torch.no_grad():
    for _ in range(1000):
        start = time.perf_counter()

        _ = model(sample)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

        latencies.append((end - start) * 1000)

latencies = np.array(latencies)

mean_latency = latencies.mean()
median_latency = np.median(latencies)
p95 = np.percentile(latencies, 95)
p99 = np.percentile(latencies, 99)

plt.figure(figsize=(10, 6))

plt.hist(latencies, bins=30, alpha=0.8)

plt.axvline(mean_latency, linestyle="--", linewidth=2,
            label=f"Mean = {mean_latency:.1f} ms")

plt.axvline(median_latency, linestyle="--", linewidth=2,
            label=f"Median = {median_latency:.1f} ms")

plt.axvline(p95, linestyle=":", linewidth=3,
            label=f"P95 = {p95:.1f} ms")

plt.axvline(p99, linestyle=":", linewidth=3,
            label=f"P99 = {p99:.1f} ms")

plt.xlabel("Inference Latency (ms)")
plt.ylabel("Count")

title_device = "GPU" if device.type == "cuda" else "CPU"

plt.title(
    f"Figure 4.6: Inference-latency distribution on the "
    f"{title_device} over one thousand single-record forward passes"
)

plt.legend()
plt.grid(alpha=0.3)

Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300)
plt.close()

print("\nSaved:", OUT_PATH)
print(f"Mean latency: {mean_latency:.2f} ms")
print(f"Median latency: {median_latency:.2f} ms")
print(f"P95 latency: {p95:.2f} ms")
print(f"P99 latency: {p99:.2f} ms")
