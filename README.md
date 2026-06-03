# Revenue Forecasting for E-Commerce: A Comparative Study of Statistical, Machine Learning, and Deep Learning Approaches

## Overview

This repository contains the complete experimental pipeline for the research article submitted to **ENIAC 2026**. The study addresses the problem of daily revenue forecasting in the online retail domain by systematically comparing three classes of predictive models across multiple granularity levels derived from a real-world transactional dataset.

The core contribution lies in the rigorous, multi-perspective evaluation: rather than predicting revenue from a single aggregated time series, we decompose the problem into three distinct series — by **Product**, by **Country**, and by **Customer** — each exhibiting different temporal dynamics, volatility profiles, and data availability windows. This design enables a nuanced analysis of model behavior under varying conditions of sparsity, seasonality, and scale.

---

## Dataset

The experiments are based on the **Online Retail II** dataset, originally published by the UCI Machine Learning Repository. It comprises approximately 1,067,371 transactions from a UK-based non-store online retailer, spanning from December 2009 to December 2010.

| Property             | Value                          |
|----------------------|--------------------------------|
| Source               | UCI Machine Learning Repository|
| Period               | Dec 2009 -- Dec 2010           |
| Raw transactions     | ~1,067,371                     |
| Target variable      | `Revenue` (daily aggregate)    |
| Granularity levels   | Product, Country, Customer     |

After cleaning (removal of cancellations, null customer IDs, and negative quantities), the dataset is aggregated into three daily time series with the following characteristics:

| Series    | Start Date   | Observations | Training Days | Validation Days |
|-----------|--------------|--------------|---------------|-----------------|
| Product   | 2010-03-15   | 194          | ~155          | ~39             |
| Country   | 2009-12-01   | 268          | ~229          | ~39             |
| Customer  | 2009-12-01   | 374          | ~335          | ~39             |

---

## Feature Engineering

A dedicated notebook (`engenharia_de_atributos.ipynb`) implements the construction of predictive features guided by SHAP-based importance analysis. All features are engineered with strict temporal discipline to prevent data leakage:

- **Expanding Window Aggregations**: Rolling statistics (mean, median, IQR) are computed using `expanding().shift(1)` to ensure that only past information is available at each time step.
- **Lag Features**: Revenue lags at 7, 14, and 30 days capture short- and medium-term autocorrelation.
- **Rolling Statistics**: 7-day and 30-day rolling mean, standard deviation, and maximum provide smoothed trend signals.
- **Calendar Variables**: Year, quarter, month, week, weekday, and day-of-year encode seasonal and cyclical patterns.
- **Pre-Christmas Indicator**: A binary flag marks the high-impact commercial period (November 1 through December 24), with associated historical mean quantities shifted by one year to avoid leakage.
- **Price Dispersion Features**: Week-weekday grouped statistics (`KnownStockCodePrice_WW_mean`, `_median`, `_std`) capture pricing volatility patterns using expanding windows.

---

## Methodology

### Validation Strategy

All models are evaluated under **Time Series Cross-Validation** using `TimeSeriesSplit` with **10 folds**. This approach creates progressively larger training windows while always validating on the immediately subsequent temporal block, preserving the causal structure of the data and providing robust performance estimates.

### Model Categories

The study evaluates models from three paradigms:

#### 1. Statistical Models (baseline)
Implemented via R (`analise_series_temporais_varejo.Rmd`):
- **AutoARIMA**: Automatic order selection for ARIMA(p,d,q) models.
- **AutoETS**: Exponential smoothing with automated component selection.
- **Prophet**: Facebook's additive regression model for time series with seasonality.

#### 2. Machine Learning Models
Implemented in `treinamento_modelos_ML.ipynb`:
- **XGBoost**: Gradient-boosted decision trees with hyperparameter optimization via Optuna (10 trials per fold). Search space includes `learning_rate` (log-uniform, 1e-3 to 0.1), `max_depth` (3--7), and `subsample` (0.6--1.0).
- **CatBoost**: Gradient boosting with native categorical support. Optimized via Optuna over `learning_rate` (log-uniform, 1e-3 to 0.1) and `depth` (4--6).

Both models use an inner validation split (last 15% of each training fold) for early stopping and hyperparameter selection. Final models are retrained on the full fold before evaluation.

#### 3. Deep Learning Models
Implemented in `treinamento_modelos_DL.ipynb` using the Darts framework:
- **BlockRNN (LSTM)**: A block-input/block-output recurrent architecture supporting `past_covariates`. Configuration: `input_chunk_length=7`, `output_chunk_length=1`, `hidden_dim=20`, `n_rnn_layers=1`, `dropout=0.1`.
- **TCN (Temporal Convolutional Network)**: Causal dilated convolutions for temporal pattern extraction. Configuration: `input_chunk_length=7`, `output_chunk_length=1`, `dropout=0.1`.

Both neural architectures receive exogenous features as `past_covariates` and undergo `MinMaxScaler` normalization. The covariates for the validation period are concatenated to the training covariates to satisfy the Darts forecasting horizon requirement.


---

## Results

The table below consolidates the performance metrics (MAE and SMAPE) obtained under the holdout evaluation protocol described in the article:

| Model      | Category         | Series   | MAE        | SMAPE (%) |
|------------|------------------|----------|------------|-----------|
| AutoARIMA  | Statistical      | Product  | 89.02      | 85.20     |
| AutoARIMA  | Statistical      | Country  | 5,730.16   | 76.91     |
| AutoARIMA  | Statistical      | Customer | 438.49     | 129.91    |
| AutoETS    | Statistical      | Product  | 77.86      | 85.01     |
| AutoETS    | Statistical      | Country  | 4,161.46   | 69.58     |
| AutoETS    | Statistical      | Customer | 454.24     | 133.34    |
| Prophet    | Statistical      | Product  | 79.91      | 84.13     |
| Prophet    | Statistical      | Country  | 5,843.69   | 73.44     |
| Prophet    | Statistical      | Customer | 451.20     | 126.17    |
| XGBoost    | Machine Learning | Product  | 110.68     | 47.72     |
| XGBoost    | Machine Learning | Country  | 20,591.53  | 166.80    |
| XGBoost    | Machine Learning | Customer | 421.14     | 112.82    |
| CatBoost   | Machine Learning | Product  | 109.84     | 47.60     |
| CatBoost   | Machine Learning | Country  | 21,530.75  | 181.50    |
| CatBoost   | Machine Learning | Customer | 417.40     | 173.10    |
| TCN        | Deep Learning    | Product  | 110.74     | 46.77     |
| TCN        | Deep Learning    | Country  | 9,328.11   | 49.44     |
| TCN        | Deep Learning    | Customer | 409.54     | 147.44    |
| LSTM       | Deep Learning    | Product  | 126.27     | 54.58     |
| LSTM       | Deep Learning    | Country  | 10,163.89  | 56.13     |
| LSTM       | Deep Learning    | Customer | 405.88     | 164.11    |

### Key Findings

- **Statistical models** (AutoETS, Prophet) achieved the lowest MAE for the Product and Customer series, suggesting that classical decomposition methods remain competitive when the underlying temporal structure is well-defined.
- **Machine Learning models** (XGBoost, CatBoost) achieved substantially lower SMAPE values for the Product series, indicating superior proportional accuracy at the item level.
- **Deep Learning models** (TCN, LSTM) demonstrated the best SMAPE performance for the Country series, where long-range dependencies and complex seasonal interactions are more pronounced.

---

## Project Structure

```
revenue_estimate/
├── README.md
├── .gitignore
└── src/
    ├── limpeza_de_dados.ipynb              # Data cleaning and preprocessing
    ├── engenharia_de_atributos.ipynb       # Feature engineering (SHAP-guided, leakage-free)
    ├── analise_series_temporais_varejo.Rmd # Statistical models (AutoARIMA, AutoETS, Prophet)
    ├── treinamento_modelos_ML.ipynb        # ML models (XGBoost, CatBoost + Optuna)
    ├── treinamento_modelos_DL.ipynb        # DL models (BlockRNN/LSTM, TCN via Darts)
    ├── requirements.txt                    # Python dependencies
    ├── data/
    │   ├── product.csv                     # Daily series aggregated by product
    │   ├── country.csv                     # Daily series aggregated by country
    │   ├── customer.csv                    # Daily series aggregated by customer
    │   └── resultados_modelos.csv          # Consolidated results across all models
    └── models/
        ├── result_json/
        │   ├── resultados_ml.json          # XGBoost and CatBoost metrics
        │   └── resultados_dl.json          # LSTM and TCN metrics
        └── result_pkl/
            ├── xgb_*.pkl                   # Serialized XGBoost models
            ├── cat_*.pkl                   # Serialized CatBoost models
            ├── lstm_*.pt                   # Serialized LSTM models (PyTorch)
            └── tcn_*.pt                    # Serialized TCN models (PyTorch)
```

---

## Reproduction

### Prerequisites

- Python 3.13+
- R 4.x (for statistical models only)
- CUDA-compatible GPU (recommended for deep learning models)

### Setup

```bash
cd src
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Execution Order

1. `limpeza_de_dados.ipynb` -- Generates cleaned transactional data.
2. `engenharia_de_atributos.ipynb` -- Produces feature-enriched datasets.
3. `analise_series_temporais_varejo.Rmd` -- Trains and evaluates statistical baselines (R environment required).
4. `treinamento_modelos_ML.ipynb` -- Trains XGBoost and CatBoost with Optuna optimization.
5. `treinamento_modelos_DL.ipynb` -- Trains BlockRNN (LSTM) and TCN via Darts.

---

## Technology Stack

| Component              | Technology                                    |
|------------------------|-----------------------------------------------|
| Data manipulation      | Pandas, NumPy                                 |
| Statistical models     | R (forecast, prophet)                         |
| ML models              | XGBoost, CatBoost                             |
| Hyperparameter tuning  | Optuna                                        |
| DL framework           | Darts (PyTorch Lightning backend)             |
| DL architectures       | BlockRNNModel (LSTM), TCNModel                |
| Scaling                | Scikit-learn (MinMaxScaler via Darts Scaler)  |
| Validation             | Scikit-learn (TimeSeriesSplit)                 |
| Visualization          | Matplotlib, Plotly                            |
| Explainability         | SHAP                                          |

---

## License

This repository is part of an academic research project submitted to ENIAC 2026. Please contact the authors before any commercial or derivative use.
