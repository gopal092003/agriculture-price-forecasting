import pickle
import numpy as np
import pandas as pd
from src.utils.save_outputs import save_csv

def run():
    with open("models/residual_models/residual_model.pkl","rb") as f:
        best_residual_model = pickle.load(f)

    with open("models/sarima/best_model.pkl","rb") as f:
        best_sarimax = pickle.load(f)

    sarimax_forecast = best_sarimax.forecast(steps=12)
    sarimax_forecast.to_csv("models/sarima/sarima_results.csv")

    residuals= pd.read_csv("models/sarima/residuals.csv")
    residuals.columns = ["date", "residual"]
    residuals["date"] = residuals.index
    residuals["residual"] = pd.to_numeric(residuals["residual"], errors="coerce")

    # Store predicted residuals
    predicted_residuals = []
    residual_series = residuals["residual"]
    # 3. Iteratively predict next 12 residuals
    for step in range(12):
        
        # Create feature row
        feature_dict = {}
        
        # Lag features (last 12 residuals)
        for lag in range(1, 13):
            feature_dict[f'lag_{lag}'] = residual_series.iloc[-lag]
        
        # Rolling features
        feature_dict['rolling_mean_3'] = residual_series.iloc[-3:].mean()
        feature_dict['rolling_std_3'] = residual_series.iloc[-3:].std()
        
        # Convert to DataFrame
        X_next = pd.DataFrame([feature_dict])
        
        # Predict residual
        next_residual = best_residual_model.predict(X_next)[0]
        
        # Store prediction
        predicted_residuals.append(next_residual)
        
        # Append to residual series for next iteration
        residual_series = pd.concat([
            residual_series,
            pd.Series([next_residual], index=[sarimax_forecast.index[step]])
        ])
    
    # Convert to Series
    predicted_residuals = pd.Series(predicted_residuals, index=sarimax_forecast.index)
    
    # 4. Final prediction = SARIMAX + residual correction
    final_forecast_log = sarimax_forecast + predicted_residuals
    final_forecast = np.expm1(final_forecast_log)
    final_forecast.to_csv("outputs/predictions/final_forecast.csv")