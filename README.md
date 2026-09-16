# Multi-Model Hybrid Framework for Water Quality Prediction

A capstone project comparing **classical machine learning**, **deep learning hybrids**, and **transformer-based hybrids** on the task of predicting water quality from physico-chemical sensor readings — both as a continuous Water Quality Index (WQI) regression and as a 3-class potability classification.

Built as part of a Capstone Project (VIT Bhopal University, 2026).

## Overview

Water quality monitoring traditionally relies on manual lab testing, which is slow and expensive. This project explores whether the physico-chemical parameters in the `water_dataX` dataset (temperature, dissolved oxygen, pH, conductivity, BOD, nitrate, and coliform counts) can be used to automate quality assessment, and — more specifically — whether increasingly complex model families (classical ML → CNN/LSTM hybrids → transformer hybrids) actually improve predictive performance on this kind of tabular sensor data.

Two tasks are evaluated in parallel:
1. **Regression** — predict a continuous Water Quality Index (WQI).
2. **Classification** — bucket samples into 3 quality tiers derived from the WQI.

## What's inside

- **10 classical models** (Linear/Logistic Regression, KNN, SVR/SVM, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, ElasticNet) evaluated with 5-fold cross-validation.
- **4 deep learning / transformer hybrid architectures**: CNN→LSTM, Conv1D + Multi-Head Attention, Transformer + LSTM, and a proposed deeper Transformer + Dense head — each evaluated with 5-fold CV and early stopping.
- A single unified preprocessing pipeline shared by every model, so comparisons are apples-to-apples.

## Dataset

`water_dataX.csv` contains water quality monitoring readings (referenced in the project report as sourced from Kaggle) for river monitoring stations in India, including:

| Column | Description |
|---|---|
| STATION CODE, LOCATIONS, STATE | Monitoring station metadata |
| Temp | Water temperature (°C) |
| D.O. (mg/l) | Dissolved oxygen |
| PH | pH level |
| CONDUCTIVITY (µmhos/cm) | Electrical conductivity |
| B.O.D. (mg/l) | Biological oxygen demand |
| NITRATENAN N+NITRITENANN (mg/l) | Nitrate + nitrite |
| FECAL COLIFORM / TOTAL COLIFORM (MPN/100ml) | Bacterial contamination indicators |
| year | Sampling year |

After cleaning, **1,991 samples × 8 numeric features** are used for modeling (station metadata and year are not used as model inputs).

> **Important methodological note:** the raw dataset does not ship with a ground-truth WQI or potability label. Both are *derived* in `load_and_preprocess_data()`: the WQI is built as a scaled average of the 8 standardized features plus Gaussian noise (to avoid one feature, like coliform counts, dominating the index), and the potability classes are created by tercile-splitting that WQI. This is a reasonable proxy for demonstrating the modeling pipeline, but it means the "ground truth" is partly synthetic — worth stating explicitly in any write-up, and worth knowing that because the noise term isn't seeded, exact metrics can drift slightly between reruns.

## Repository structure

```
water-quality-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── water_dataX.csv
└── notebooks/
    └── unifiedmodel.ipynb
```

## Setup

```bash
git clone <your-repo-url>
cd water-quality-prediction
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
jupyter notebook notebooks/unifiedmodel.ipynb
```

Run all cells top to bottom. The notebook loads `data/water_dataX.csv` by default (relative path `water_dataX.csv` — adjust the path in `load_and_preprocess_data()` if you keep the data elsewhere), trains every model with 5-fold cross-validation, and prints a final comparison table at the end.

## Models implemented

**Classical (regression + classification variants, 10 each):**
Linear/Logistic Regression, KNN, SVR/SVM, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, ElasticNet

**Deep learning & transformer hybrids (4):**
- `CNN_LSTM` — stacked Conv1D → LSTM
- `Conv_Attention` — Conv1D + Multi-Head Attention
- `Transformer_LSTM` — Transformer encoder block → LSTM
- `Proposed_Deep_Transformer` — 2-layer deeper Transformer encoder + dense head

## Results

These are the actual 5-fold cross-validated results produced by the notebook (not projected figures):

**Classical regression (predicting WQI)**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Linear Regression** | 3.18 ± 0.04 | 3.99 ± 0.06 | **0.45 ± 0.11** |
| Gradient Boosting | 3.37 ± 0.08 | 4.51 ± 0.42 | 0.31 ± 0.08 |
| Random Forest | 3.45 ± 0.08 | 4.55 ± 0.38 | 0.30 ± 0.08 |
| ElasticNet | 3.49 ± 0.12 | 4.54 ± 0.20 | 0.29 ± 0.12 |
| CatBoost | 3.47 ± 0.09 | 4.72 ± 0.56 | 0.25 ± 0.08 |
| SVR | 3.40 ± 0.12 | 4.90 ± 0.59 | 0.20 ± 0.02 |
| KNN | 3.61 ± 0.07 | 4.80 ± 0.36 | 0.22 ± 0.08 |
| LightGBM | 3.69 ± 0.16 | 5.00 ± 0.48 | 0.15 ± 0.11 |
| XGBoost | 3.77 ± 0.14 | 5.13 ± 0.33 | 0.10 ± 0.12 |
| Decision Tree | 4.73 ± 0.25 | 6.06 ± 0.28 | -0.27 ± 0.26 |

**Classical classification (potability tier)**

| Model | Precision % | Recall % | F1 % |
|---|---|---|---|
| Logistic Regression | 45.7 ± 2.2 | 44.9 ± 2.0 | **45.0 ± 2.1** |
| ElasticNet (class) | 45.7 ± 2.3 | 44.8 ± 2.2 | 45.0 ± 2.2 |
| SVM | 49.7 ± 1.8 | 44.9 ± 1.3 | 44.6 ± 1.2 |
| Gradient Boosting | 42.0 ± 0.7 | 41.8 ± 0.7 | 41.8 ± 0.6 |
| CatBoost | 40.8 ± 1.4 | 40.8 ± 1.5 | 40.7 ± 1.5 |
| LightGBM | 40.1 ± 1.9 | 40.2 ± 1.9 | 40.1 ± 1.9 |
| Random Forest | 40.0 ± 1.4 | 39.9 ± 1.4 | 39.9 ± 1.4 |
| XGBoost | 39.9 ± 1.0 | 39.8 ± 1.0 | 39.8 ± 1.0 |
| KNN | 41.0 ± 2.0 | 40.2 ± 2.0 | 39.6 ± 2.0 |
| Decision Tree | 37.8 ± 1.7 | 37.7 ± 1.7 | 37.7 ± 1.7 |

**Deep learning & transformer hybrids (multi-task: regression + classification)**

| Model | MAE | RMSE | R² | Precision % | Recall % | F1 % |
|---|---|---|---|---|---|---|
| Conv_Attention | 3.30 ± 0.15 | 4.20 ± 0.22 | **0.40 ± 0.09** | 10.8 ± 1.0 | 33.3 ± 0.0 | 16.3 ± 1.2 |
| Proposed_Deep_Transformer | 3.32 ± 0.11 | 4.27 ± 0.27 | 0.38 ± 0.07 | **51.8 ± 12.0** | 40.1 ± 3.1 | **32.2 ± 5.7** |
| Transformer_LSTM | 3.28 ± 0.08 | 4.43 ± 0.51 | 0.34 ± 0.03 | 30.3 ± 11.2 | 38.3 ± 4.1 | 25.6 ± 7.8 |
| CNN_LSTM | 3.71 ± 0.25 | 5.29 ± 0.75 | 0.06 ± 0.12 | 10.8 ± 0.7 | 33.3 ± 0.0 | 16.3 ± 0.8 |

**Takeaway from the actual run:** on this dataset and target formulation, the simple classical baselines (Linear Regression for WQI, Logistic Regression for potability) hold up better than the deeper hybrid architectures, though the proposed Transformer does clearly win on classification F1 among the neural models. This is a useful, honest finding in itself — it suggests the added architectural complexity isn't paying for itself here, likely because the feature set is small (8 features) and largely non-sequential, which favors simpler models over attention/recurrent mechanisms designed for richer or more temporal input.

## Team

Shreyansh Lohumi · Vansh Dewan · Rudhinandan Patel · Ankit Kumar Jha · Soumyadeep Nath
Supervisor: Dr. Vivek Jain — VIT Bhopal University

## License

Add a license of your choice (e.g., MIT) if you want others to freely reuse this code.