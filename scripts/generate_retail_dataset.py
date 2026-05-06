import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

OUT = Path("data/generated/products_structured.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

categories = {
    "Electronics": {"price": (1200, 80), "stock": (65, 20), "warranty": (30, 5), "rating": (4.15, 0.28)},
    "Clothing": {"price": (80, 12), "stock": (220, 55), "warranty": (3, 2), "rating": (3.85, 0.35)},
    "Books": {"price": (15, 4), "stock": (180, 45), "warranty": (0.5, 0.8), "rating": (4.35, 0.25)},
    "Home": {"price": (250, 35), "stock": (120, 35), "warranty": (14, 4), "rating": (3.95, 0.30)},
    "Sports": {"price": (150, 25), "stock": (145, 40), "warranty": (9, 3), "rating": (4.05, 0.30)},
    "Toys": {"price": (35, 8), "stock": (260, 60), "warranty": (2, 1.5), "rating": (4.10, 0.35)},
    "Beauty": {"price": (55, 10), "stock": (210, 50), "warranty": (1, 1), "rating": (4.45, 0.22)},
    "Automotive": {"price": (600, 60), "stock": (80, 25), "warranty": (22, 5), "rating": (3.75, 0.32)},
}

rows = []
n_per_class = 1250
pid = 1

for category, params in categories.items():
    for i in range(n_per_class):
        price = max(5, np.random.normal(*params["price"]))
        stock = int(np.clip(np.random.normal(*params["stock"]), 0, 500))
        warranty = int(np.clip(np.random.normal(*params["warranty"]), 0, 36))
        rating = float(np.clip(np.random.normal(*params["rating"]), 1.0, 5.0))

        rows.append({
            "Product ID": f"PROD-{pid:05d}",
            "Product Name": f"{category} Item {i+1}",
            "Product Category": category,
            "Product Description": f"Synthetic {category.lower()} retail product",
            "Price": round(price, 2),
            "Stock Quantity": stock,
            "Warranty Period": warranty,
            "Product Dimensions": "Standard",
            "Manufacturing Date": "2025-01-01",
            "Expiration Date": "2027-01-01",
            "SKU": f"SKU-{pid:05d}",
            "Product Tags": category.lower(),
            "Color/Size Variations": "Standard",
            "Product Ratings": round(rating, 3),
        })
        pid += 1

df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv(OUT, index=False)

print(f"Saved {OUT}")
print(df.shape)
print(df["Product Category"].value_counts())
print(df.groupby("Product Category")[["Price", "Stock Quantity", "Warranty Period", "Product Ratings"]].mean())
