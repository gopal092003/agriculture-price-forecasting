import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
from itertools import product

def train_sarima(df):
    # Define ranges
    p = range(0, 4)   # 0–3
    d = range(0, 3)   # 0–2
    q = range(0, 4)   # 0–3
    
    P = range(0, 3)   # 0–2
    D = range(0, 2)   # 0–1
    Q = range(0, 3)   # 0–2
    
    # Generate combinations
    a = np.array(list(product(p, d, q)))   # (p,d,q)
    A = np.array(list(product(P, D, Q)))   # (P,D,Q)

    # Initialize trackers
    best_aic = float('inf')
    best_order = None
    best_seasonal_order = None
    best_model = None
    
    # Loop over all combinations
    for i in range(len(a)):
        for j in range(len(A)):
            p, d, q = a[i]
            P, D, Q = A[j]
    
            try:
                model = SARIMAX(
                    df['price'],
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, 12),  # monthly seasonality
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(disp=False)
    
                # Check if this model is better
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_order = (p, d, q)
                    best_seasonal_order = (P, D, Q, 12)
                    best_model = model
    
            except Exception as e:
                continue
    
    # Assign best model
    best_sarimax = best_model

    reports={}
    reports['SARIMA']={}
    reports['SARIMA']["Best SARIMAX model"]={
        "Order (p,d,q)": f"{best_order}",    
        "Seasonal order (P,D,Q,s)": f"{best_seasonal_order}",
        "AIC": f"{best_aic}"
    }
    return (best_model, reports, df)