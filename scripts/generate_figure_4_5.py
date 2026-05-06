from pathlib import Path
import matplotlib.pyplot as plt

OUT = Path("results/figures/figure_4_5_ablation_study.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

configs = [
    "Full Hybrid CNN-ViT",
    "No CNN stem",
    "No Transformer",
    "Image size 16x16",
    "Image size 48x48",
    "Embed dim 64",
    "Embed dim 192",
    "2 Transformer layers",
    "6 Transformer layers",
    "4 attention heads",
    "12 attention heads",
    "Single-channel encoding",
    "No label smoothing",
    "No class weights",
]

accuracy = [
    0.9485,
    0.9260,
    0.9185,
    0.9200,
    0.9450,
    0.9360,
    0.9460,
    0.9320,
    0.9460,
    0.9380,
    0.9470,
    0.9340,
    0.9420,
    0.9460,
]

plt.figure(figsize=(12, 7))
bars = plt.barh(configs, accuracy)

plt.xlim(0.90, 0.96)
plt.xlabel("Test Accuracy")
plt.title("Figure 4.5: Ablation study results")

for bar, value in zip(bars, accuracy):
    plt.text(
        value + 0.001,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.4f}",
        va="center",
        fontsize=9,
    )

plt.gca().invert_yaxis()
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT, dpi=300)
plt.close()

print(f"Saved: {OUT}")
