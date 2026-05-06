"""
dashboard.py
=============
Streamlit dashboard for the Hybrid CNN-ViT retail framework.

Visualises:
    - Inventory KPIs
    - Stock-level distribution
    - Rating-class breakdown
    - Per-category counts
    - CNN-ViT vs baseline metrics
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Hybrid CNN-ViT Retail Analytics",
    layout="wide",
)

st.title("📊 Hybrid CNN-ViT Retail Analytics")
st.caption("Real-time inventory insights powered by a Hybrid CNN-Vision-Transformer model.")
st.markdown("---")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
DATA_FILE = "data/products.csv"
METRICS_FILE = "runs/cnn_vit/test_metrics.json"


@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    df = pd.read_csv(path)

    if "Product Ratings" in df.columns:
        df["Rating Class"] = df["Product Ratings"].apply(
            lambda x: "High-Rated" if x >= 4.0 else "Low-Rated"
        )
    if "Stock Quantity" in df.columns:
        def lvl(q):
            if q > 30:
                return "High"
            if q >= 10:
                return "Medium"
            return "Low"
        df["Stock Level"] = df["Stock Quantity"].apply(lvl)
    return df


@st.cache_data
def load_metrics(path: str) -> dict:
    if not Path(path).exists():
        return {}
    with open(path) as f:
        return json.load(f)


df = load_dataset(DATA_FILE)
metrics = load_metrics(METRICS_FILE)


# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
if df.empty:
    st.error(f"❌ Could not load dataset from {DATA_FILE}")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("📦 Total Products", f"{len(df):,}")
if "Price" in df.columns:
    c2.metric("💰 Avg Price", f"${df['Price'].mean():.2f}")
if "Rating Class" in df.columns:
    pct = (df["Rating Class"] == "High-Rated").mean() * 100
    c3.metric("⭐ High-Rated %", f"{pct:.1f}%")
if "Stock Quantity" in df.columns:
    c4.metric("📊 Total Stock", f"{int(df['Stock Quantity'].sum()):,}")

st.markdown("---")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    if "Product Category" in df.columns:
        cat = df["Product Category"].value_counts().reset_index()
        cat.columns = ["Category", "Count"]
        fig = px.bar(
            cat, x="Category", y="Count",
            title="Products by Category",
            color="Count", color_continuous_scale="Blues",
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    if "Stock Level" in df.columns:
        sl = df["Stock Level"].value_counts().reset_index()
        sl.columns = ["Level", "Count"]
        fig = px.bar(
            sl, x="Level", y="Count",
            title="Stock Level Distribution",
            color="Level",
            color_discrete_map={
                "High": "#27ae60", "Medium": "#f39c12", "Low": "#e74c3c"
            },
        )
        st.plotly_chart(fig, use_container_width=True)


if "Stock Quantity" in df.columns:
    fig = px.histogram(
        df, x="Stock Quantity", nbins=30,
        title="Stock Quantity Distribution",
        color_discrete_sequence=["#667eea"],
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Model metrics
# ---------------------------------------------------------------------------
if metrics:
    st.subheader("🤖 Hybrid CNN-ViT Model Performance")
    a, b, c, d = st.columns(4)
    a.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
    b.metric("Macro F1", f"{metrics.get('macro_f1', 0)*100:.2f}%")
    c.metric("Macro Precision", f"{metrics.get('macro_precision', 0)*100:.2f}%")
    d.metric("Macro ROC-AUC", f"{metrics.get('macro_roc_auc', 0)*100:.2f}%")

    if "confusion_matrix" in metrics:
        cm = metrics["confusion_matrix"]
        cm_df = pd.DataFrame(cm)
        fig = px.imshow(
            cm_df,
            text_auto=True,
            title="Confusion Matrix",
            color_continuous_scale="Blues",
            aspect="auto",
            labels=dict(x="Predicted", y="True"),
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Inventory table
# ---------------------------------------------------------------------------
st.subheader("📋 Inventory Details")
display = [c for c in [
    "Product ID", "Product Name", "Product Category",
    "Price", "Stock Quantity", "Warranty Period",
    "Product Ratings", "Rating Class", "Stock Level"
] if c in df.columns]
st.dataframe(df[display], use_container_width=True)
st.download_button(
    "📥 Export to CSV",
    df.to_csv(index=False).encode("utf-8"),
    "inventory_export.csv",
    "text/csv",
)
