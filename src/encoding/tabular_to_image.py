# src/encoding/tabular_to_image.py
import numpy as np
from typing import Dict, Tuple
import math

class CorrelationAwareEncoder:
    def __init__(self, image_size: int = 32, num_channels: int = 3, corr_method: str = "pearson"):
        self.image_size = image_size
        self.num_channels = num_channels
        self.corr_method = corr_method
        self.feature_to_pixel: Dict[int, Tuple[int, int]] = {}
        self.feature_order = None

    def fit(self, X: np.ndarray) -> "CorrelationAwareEncoder":
        """Fit deterministic mapping using correlation."""
        n_features = X.shape[1]
        if n_features > self.image_size ** 2:
            raise ValueError("Too many features for grid")

        # Compute correlation matrix
        if self.corr_method == "pearson":
            corr = np.abs(np.corrcoef(X.T))
        else:  # spearman
            corr = np.abs(np.corrcoef(np.apply_along_axis(lambda x: np.argsort(np.argsort(x)), 0, X).T))
        
        np.fill_diagonal(corr, 0.0)
        
        # Importance + ordering
        importance = corr.sum(axis=1)
        self.feature_order = np.argsort(-importance).tolist()

        # Simple but effective placement: row-major (as in thesis)
        # TODO (future): Use graph layout / TSP for better clustering of correlated groups
        grid_positions = [(i, j) for i in range(self.image_size) 
                         for j in range(self.image_size)]
        
        for idx, feat_idx in enumerate(self.feature_order):
            self.feature_to_pixel[feat_idx] = grid_positions[idx]
        
        return self

    def transform_row(self, x: np.ndarray) -> np.ndarray:
        """Convert one sample to 3-channel image."""
        img = np.zeros((self.num_channels, self.image_size, self.image_size), dtype=np.float32)
        
        for feat_idx, (i, j) in self.feature_to_pixel.items():
            v = float(x[feat_idx])
            img[0, i, j] = np.tanh(v)
            img[1, i, j] = 1.0 / (1.0 + np.exp(-v))
            img[2, i, j] = np.sign(v) * np.log1p(abs(v))
        
        return img

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Batch transform."""
        return np.stack([self.transform_row(row) for row in X])