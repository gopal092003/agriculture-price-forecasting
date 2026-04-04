# 📊 Price Forecasting Project Documentation

## 📌 Overview

This project implements a **hybrid time series forecasting pipeline** that combines:

* **Statistical modeling** (SARIMA)
* **Machine Learning** (Residual modeling using XGBoost / Gradient Boosting)

The system forecasts future prices by:

1. Modeling the main trend using SARIMA
2. Learning residual patterns using ML models
3. Combining both to produce final predictions

---

## 🎯 Objectives

* Analyze historical price data
* Identify trend, seasonality, and volatility
* Build a robust forecasting pipeline
* Improve SARIMA forecasts using residual learning
* Generate interpretable visualizations and reports

---

## 🏗️ Project Structure

```
price-forecasting-project/
│
├── config/                # Configuration files
├── data/                  # Raw → processed datasets
├── models/                # Trained models
├── outputs/               # Plots, reports, predictions
├── pipeline/              # Pipeline scripts
├── src/                   # Core modules
├── notebooks/             # Experiments
├── requirements.txt
├── README.md
├── documentation.md
```

---

## ⚙️ Pipeline Workflow

### 🔹 1. Data Processing & Analysis

**File:** `pipeline/run_analysis.py`

Steps:

* Load raw data
* Clean missing values and duplicates
* Perform EDA:

  * Distribution analysis
  * Monthly patterns
  * Seasonality
  * Trend analysis
  * Volatility
  * Outlier detection
* Feature engineering
* Save processed dataset

---

### 🔹 2. Model Training

**File:** `pipeline/train_models.py`

#### 🧠 SARIMA Model

* Grid search over `(p,d,q)` and `(P,D,Q,s)`
* Select best model using **AIC**

#### 🔁 Residual Modeling

* Extract SARIMA residuals
* Create features:

  * Lag features (1–12)
  * Rolling statistics
* Train models:

  * XGBoost
  * Gradient Boosting
* Select best model based on **RMSE**

---

### 🔹 3. Forecast Generation

**File:** `pipeline/generate_forecast.py`

Steps:

1. Generate SARIMA forecast
2. Predict residuals iteratively
3. Combine:

```
Final Forecast = SARIMA Forecast + Residual Prediction
```

4. Apply inverse transformation
5. Save predictions

---

### 🔹 4. Full Pipeline Execution

**File:** `pipeline/run_pipeline.py`

```python
run_analysis()
train()
forecast()
```

---

## 📊 Feature Engineering

Features include:

* Log transformation of price
* Lag features: `lag_1`, `lag_2`, `lag_3`, `lag_12`
* Rolling statistics:

  * Mean (3 months)
  * Std deviation (3 months)
* Trend:

  * First difference
* Seasonality:

  * Sin/Cos month encoding
* Recency weighting

---

## 📈 Outputs

### 📊 Plots

Saved in:

```
outputs/plots/
```

Includes:

* EDA visualizations
* Seasonality plots
* Trend analysis
* Volatility charts
* Residual plots

---

### 📄 Reports

Saved in:

```
outputs/reports/
```

Includes:

* Dataset overview
* Stationarity tests (ADF)
* Seasonal variation
* Volatility summary
* Model performance

---

### 🔮 Predictions

Saved in:

```
outputs/predictions/final_forecast.csv
```

---

## 🧠 Models Used

### 1. SARIMA

Captures:

* Trend
* Seasonality
* Autocorrelation

### 2. Residual Models

Captures:

* Non-linear patterns
* Short-term dependencies

Models:

* XGBoost
* Gradient Boosting

---

## ⚡ Key Concepts

### 🔹 Hybrid Modeling

Combines:

* Statistical + Machine Learning

### 🔹 Residual Learning

Instead of predicting full series:

```
Residual = Actual - SARIMA Prediction
```

ML model learns residuals to improve accuracy.

---

## 📉 Evaluation Metrics

* RMSE (Root Mean Squared Error)
* (Recommended additions)

  * MAE
  * MAPE

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run pipeline

```bash
python -m pipeline.run_pipeline
```

---

## ⚠️ Assumptions & Limitations

* Data is monthly and continuous
* No external variables (exogenous features)
* SARIMA grid search can be slow
* Residual model depends on SARIMA quality

---

## 🔧 Possible Improvements

* Add **walk-forward validation**
* Use **auto_arima** for faster tuning
* Add **confidence intervals**
* Incorporate external features (weather, demand, etc.)
* Replace with deep learning (LSTM / Transformer)

---

## 📌 Key Highlights

✔ Modular architecture
✔ Config-driven pipeline
✔ Hybrid forecasting approach
✔ Automated EDA + reporting
✔ Scalable and extensible

---

## 🧾 Conclusion

This project demonstrates a **robust end-to-end time series forecasting system** that:

* Leverages both statistical and machine learning techniques
* Produces interpretable insights
* Improves forecast accuracy using residual modeling

---

## 👨‍💻 Author

**Gopal Gupta**

---

## 📜 License

This project is for educational and research purposes.
