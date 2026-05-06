import numpy as np
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt

class CorrelationAwareEncoder:
    def __init__(self, image_size: int = 32, num_channels: int = 3, corr_method: str = "pearson"):
        self.image_size = image_size
        self.num_channels = num_channels
        self.corr_method = corr_method
        self.feature_to_pixel: Dict[int, Tuple[int, int]] = {}
        self.feature_names: List[str] = None
        self.feature_order = None
        self.corr_matrix = None

    def fit(self, X: np.ndarray, feature_names: List[str] = None) -> "CorrelationAwareEncoder":
        n_features = X.shape[1]
        self.feature_names = feature_names or [f"feat_{i}" for i in range(n_features)]

        # Correlation matrix
        if self.corr_method == "pearson":
            corr = np.corrcoef(X.T)
        else:
            ranks = np.apply_along_axis(lambda x: np.argsort(np.argsort(x)), 0, X)
            corr = np.corrcoef(ranks.T)
        
        self.corr_matrix = np.abs(corr)
        np.fill_diagonal(self.corr_matrix, 0)

        importance = self.corr_matrix.sum(axis=1)
        self.feature_order = np.argsort(-importance).tolist()

        # Place on grid
        grid_positions = [(i, j) for i in range(self.image_size) for j in range(self.image_size)]
        for idx, feat_idx in enumerate(self.feature_order):
            self.feature_to_pixel[feat_idx] = grid_positions[idx]

        return self

    def transform_row(self, x: np.ndarray) -> np.ndarray:
        img = np.zeros((self.num_channels, self.image_size, self.image_size), dtype=np.float32)
        for feat_idx, (i, j) in self.feature_to_pixel.items():
            v = float(x[feat_idx])
            img[0, i, j] = np.tanh(v)
            img[1, i, j] = 1.0 / (1.0 + np.exp(-v))
            img[2, i, j] = np.sign(v) * np.log1p(abs(v))
        return img

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.stack([self.transform_row(row) for row in X])

    def visualize_placement(self, save_path: str = "results/figures/feature_placement.png"):
        plt.figure(figsize=(10, 10))
        plt.imshow(np.zeros((self.image_size, self.image_size)), cmap='gray', alpha=0.3)
        for feat_idx, (i, j) in self.feature_to_pixel.items():
            name = self.feature_names[feat_idx] if self.feature_names else f"F{feat_idx}"
            plt.text(j, i, name, ha='center', va='center', fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.8))
        plt.title("Correlation-Aware Feature Placement")
        plt.axis('off')
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Placement saved: {save_path}")
