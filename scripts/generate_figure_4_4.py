from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("results/figures/figure_4_4_macro_comparison.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

models = [
    "Random Forest",
    "XGBoost",
    "CNN-only",
    "ViT-only",
    "Hybrid CNN-ViT"
]

accuracy = [0.889, 0.918, 0.926, 0.935, 0.9485]
macro_f1 = [0.884, 0.913, 0.921, 0.931, 0.9482]
precision = [0.887, 0.916, 0.924, 0.933, 0.9490]
recall = [0.884, 0.913, 0.921, 0.931, 0.9478]
auc = [0.943, 0.966, 0.972, 0.981, 0.997]

x = np.arange(len(models))
width = 0.15

plt.figure(figsize=(12, 6))

plt.bar(x - 2*width, accuracy, width, label="Accuracy")
plt.bar(x - width, macro_f1, width, label="Macro F1")
plt.bar(x, precision, width, label="Macro Precision")
plt.bar(x + width, recall, width, label="Macro Recall")
plt.bar(x + 2*width, auc, width, label="Macro AUC")

plt.xticks(x, models, rotation=20, ha="right")
plt.ylim(0.84, 1.02)
plt.ylabel("Score")
plt.title("Figure 4.4: Macro performance comparison across models")
plt.legend()
plt.grid(axis="y", alpha=0.3)

for i, v in enumerate(accuracy):
    if models[i] == "Hybrid CNN-ViT":
        plt.text(i - 2*width, v + 0.006, f"{v:.3f}", ha="center", fontsize=9)
        plt.text(i - width, macro_f1[i] + 0.006, f"{macro_f1[i]:.3f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(OUT, dpi=300)
plt.close()

print(f"Saved: {OUT}")
