# energy-consumption-time-series-forecasting

# ⚡ Energy Consumption Forecasting with Machine Learning

This project focuses on **forecasting energy consumption** using time series analysis and machine learning. We leverage historical energy demand data (from **2002 to 2017**) and apply **feature engineering**, **seasonality detection**, and an **XGBoost regression model** to make accurate predictions of energy usage.

## 📊 Dataset Overview

The dataset spans 15 years and shows strong evidence of **seasonality**—repeating patterns influenced by time-based factors like month, quarter, and hour of the day.

### Patterns Considered

While analyzing time-based patterns, several possibilities were considered:

- Pure randomness (no trend)
- Quadratic (Curvilinear) trends
- Increasing linear trends
- Seasonal patterns
- Seasonal patterns combined with a linear trend

The final data revealed **seasonal behavior**, making time-series modeling a suitable approach.

## 🧠 Machine Learning Pipeline

We use a time-based cross-validation strategy (`TimeSeriesSplit`) and train an **XGBoost regressor** to learn from engineered time features.

### Feature Engineering

To capture temporal patterns, we create the following features:

- `dayofyear`, `hour`, `dayofweek`, `quarter`, `month`, `year`
- `lag1`, `lag2`, `lag3` – past values to capture short-term trends

### Model Training

We apply a **rolling validation** approach using `TimeSeriesSplit`:

```python
tss = TimeSeriesSplit(n_splits=5, test_size=24*365*1, gap=24)

```

### 🧪 Cross-Validation Strategy

Each fold includes:

- ✅ Training on historical data  
- 🔍 Testing on unseen future data  
- 📉 Evaluation using Root Mean Squared Error (RMSE)

### 🛠️ Model Details

We use the `XGBRegressor` with the following configuration:

```python
xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=3,
    learning_rate=0.01,
    early_stopping_rounds=50,
    objective='reg:linear'
)
```

## 📈 Results

The model produces reliable forecasts with good generalization across different time windows. It captures both short-term lags and long-term seasonality, demonstrating the strength of combining **time-aware cross-validation** with **gradient boosting**.
