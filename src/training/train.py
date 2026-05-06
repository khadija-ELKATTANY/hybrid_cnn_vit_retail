import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class TabularToImageEncoder:
    def __init__(self, grid_size=32):
        self.grid_size = grid_size
        self.scaler = StandardScaler()
        
    def fit(self, X: pd.DataFrame):
        """Fit scaler on numeric features"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        self.scaler.fit(X[numeric_cols])
        return self
    
    def transform(self, X: pd.DataFrame):
        """Convert tabular rows to fixed-size images"""
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_scaled = self.scaler.transform(X[numeric_cols])
        
        batch_size = X_scaled.shape[0]
        images = np.zeros((batch_size, 3, self.grid_size, self.grid_size), dtype=np.float32)
        
        for i in range(batch_size):
            flat = X_scaled[i]
            n_features = len(flat)
            
            # Create a square grid by repeating/padding features
            side = self.grid_size
            grid = np.zeros((side, side), dtype=np.float32)
            
            # Fill the grid (repeat features to fill the space)
            idx = 0
            for row in range(side):
                for col in range(side):
                    grid[row, col] = flat[idx % n_features]
                    idx += 1
            
            # 3 channels as in thesis
            images[i, 0] = np.tanh(grid)                    # Channel 1
            images[i, 1] = 1 / (1 + np.exp(-grid))         # Channel 2 - sigmoid
            images[i, 2] = np.sign(grid) * np.log1p(np.abs(grid))  # Channel 3
        
        return images