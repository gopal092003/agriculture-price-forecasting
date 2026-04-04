import pandas as pd
import numpy as np
import os

from src.features.build_features import feature_engineering
from src.models.residual_model import residual_model
from src.models.sarima import train_sarima
from src.utils.save_outputs import save_model, save_csv, save_json
from src.utils.plotting import save_plot
from src.config.config import load_config

def run():
    # Load configs
    paths_config = load_config("config/paths_config.yaml")
    model_config = load_config("config/model_config.yaml")

    # Extract paths
    final_path = paths_config["data"]["final"]
    residual_model_path = paths_config["models"]["residual_model"]
    sarima_model_path = paths_config["models"]["sarima"]
    residual_plot_path = paths_config["outputs"]["plots"]
    reports_base = paths_config["outputs"]["reports"]

    # -------------------------------
    # DATA Import
    # -------------------------------
    df = pd.read_csv(final_path, parse_dates=["date"])

    # -------------------------------
    # Feature Engineering
    # -------------------------------
    df = feature_engineering(df)

    # ------------------------------
    # SARIMA parameters
    # ------------------------------

    best_sarima_model, reports, df = train_sarima(df)
    save_model(best_sarima_model, sarima_model_path)  

    # 1. Get residuals to feed to XGBoost / LightGBM
    residuals = best_sarima_model.resid.dropna()
    
    # 2. Prepare features for boosting (e.g., lag features)
    df_ml = pd.DataFrame({
        'residual': residuals
    })
    for lag in range(1, 13):
        df_ml[f'lag_{lag}'] = df_ml['residual'].shift(lag)
    
    df_ml['rolling_mean_3'] = df_ml['residual'].rolling(3).mean()
    df_ml['rolling_std_3'] = df_ml['residual'].rolling(3).std()
    
    df_ml = df_ml.dropna()

    residuals_df = residuals.to_frame(name="residual")
    residuals_df["date"] = residuals_df.index
    residuals_df.to_csv("models/sarima/residuals.csv", index=False)

    fig = residuals.to_frame().plot(title="SARIMA Residuals").get_figure()
    save_plot(fig, os.path.join(residual_plot_path, "residuals/residual.png"))

    # -------------------------------
    # TRAIN SPLIT
    # -------------------------------
    test_size = model_config["training"]["test_size"]
    train_size = int(len(df_ml) * (1 - test_size))

    train = df_ml[:train_size]
    test = df_ml[train_size:]
    
    X_train = train.drop('residual', axis=1)
    y_train = train['residual']
    
    X_test = test.drop('residual', axis=1)
    y_test = test['residual']
    
    indices = np.arange(1, len(y_train)+1)
    recency_weight = np.log(indices + 1)
    error_weight = np.abs(y_train)
    
    weights = recency_weight * (1 + error_weight)

    # -------------------------------
    # MODEL TRAINING
    # -------------------------------
    model,reports['residual model'] = residual_model(X_train, y_train, X_test, y_test, weights)

    # -------------------------------
    # SAVE MODEL
    # -------------------------------
    save_model(model, residual_model_path)
    file_path = os.path.join(reports_base, "train reports.json")
    
    save_json(reports, file_path)