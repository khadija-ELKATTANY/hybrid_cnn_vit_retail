# hybrid_cnn_vit_retail
hybrid-cnn-vit-retail/
├── README.md
├── LICENSE
├── requirements.txt
├── config.py
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── dataset.csv                  # your 10k dataset
│
├── src/
│   ├── encoding/
│   │   ├── __init__.py
│   │   ├── tabular_to_image.py      # Core innovation
│   │   └── test_encoder.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── hybrid_cnn_vit.py        # Main model
│   │   └── test_model.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   └── trainer.py
│   │
│   ├── inference/
│   │   ├── kafka_consumer.py
│   │   └── test_inference.py
│   │
│   └── utils/
│       ├── metrics.py
│       ├── visualizations.py
│       └── common.py
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Encoding_Examples.ipynb
│   └── 03_Training_Analysis.ipynb
│
├── dashboard/
│   └── app.py                       # Streamlit
│
├── scripts/
│   ├── run_training.sh
│   ├── run_producer.py
│   └── run_consumer.sh
│
├── results/
│   ├── models/                      # best_model.pth
│   ├── figures/
│   └── logs/
│
└── tests/
    └── test_pipeline.py