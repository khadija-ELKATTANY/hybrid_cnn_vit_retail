import os
import sys
import json
import time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from src.encoding.tabular_to_image import CorrelationAwareEncoder
from src.models.hybrid_cnn_vit import HybridCNNViT


class Config:
    DATA_PATH = "data/generated/products_structured.csv"
    IMAGE_SIZE = 32
    EPOCHS = 20
    BATCH_SIZE = 64
    LR = 0.001
    OUTPUT_DIR = "results"


def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            outputs = model(xb)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, macro_f1, all_labels, all_preds, all_probs


def main():
    config = Config()
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(f"{config.OUTPUT_DIR}/figures").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(config.DATA_PATH)

    feature_cols = ["Price", "Stock Quantity", "Warranty Period", "Product Ratings"]
    target_col = "Product Category"

    X = df[feature_cols].values
    y_raw = df[target_col].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    encoder = CorrelationAwareEncoder(image_size=config.IMAGE_SIZE)
    encoder.fit(X_train, feature_names=feature_cols)
    encoder.visualize_placement()

    X_train_img = encoder.transform(X_train)
    X_test_img = encoder.transform(X_test)

    train_ds = TensorDataset(torch.tensor(X_train_img).float(), torch.tensor(y_train).long())
    test_ds = TensorDataset(torch.tensor(X_test_img).float(), torch.tensor(y_test).long())

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridCNNViT(num_classes=len(label_encoder.classes_)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)

    history = {
        "train_loss": [],
        "test_accuracy": [],
        "test_macro_f1": []
    }

    print("Starting training...")
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0
        total_items = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            total_items += xb.size(0)

        avg_loss = total_loss / total_items
        test_acc, test_f1, _, _, _ = evaluate_model(model, test_loader, device)

        history["train_loss"].append(avg_loss)
        history["test_accuracy"].append(test_acc)
        history["test_macro_f1"].append(test_f1)

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | "
            f"loss={avg_loss:.4f} | "
            f"test_acc={test_acc:.4f} | "
            f"macro_f1={test_f1:.4f}"
        )

    final_acc, final_f1, labels, preds, probs = evaluate_model(model, test_loader, device)

    metrics = {
        "accuracy": final_acc,
        "macro_f1": final_f1,
        "class_names": label_encoder.classes_.tolist(),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "classification_report": classification_report(
            labels,
            preds,
            target_names=label_encoder.classes_,
            output_dict=True,
            zero_division=0
        )
    }

    with open(f"{config.OUTPUT_DIR}/test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(f"{config.OUTPUT_DIR}/history.json", "w") as f:
        json.dump(history, f, indent=2)

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_DIR}/figures/confusion_matrix.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["test_accuracy"], label="Test Accuracy")
    plt.plot(history["test_macro_f1"], label="Macro F1")
    plt.xlabel("Epoch")
    plt.title("Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_DIR}/figures/training_history.png")
    plt.close()

    start = time.perf_counter()
    sample_x, _ = test_ds[0]
    sample_x = sample_x.unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(100):
            _ = model(sample_x)
    avg_latency_ms = ((time.perf_counter() - start) / 100) * 1000

    latency = {"single_record_latency_ms": avg_latency_ms}
    with open(f"{config.OUTPUT_DIR}/latency.json", "w") as f:
        json.dump(latency, f, indent=2)

    torch.save(model.state_dict(), f"{config.OUTPUT_DIR}/model_final.pth")

    print("Training finished.")
    print(f"Final accuracy: {final_acc:.4f}")
    print(f"Final macro F1: {final_f1:.4f}")
    print(f"Latency: {avg_latency_ms:.2f} ms")
    print(f"Saved outputs to: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
