import numpy as np
import pandas as pd
from src.features.transformations import log_transform

def build_features(df):
    df = df.copy()
    df = df.set_index("date").asfreq("MS")

    df["price"] = np.log1p(df["avg_monthly_price"])

    df['year']       = df.index.year
    df['month']      = df.index.month
    df['month_name'] = df.index.strftime('%b')
    df['quarter']    = df.index.quarter
    
    season_map = {12:'Winter', 1:'Winter', 2:'Winter',
                  3:'Summer', 4:'Summer', 5:'Summer',
                  6:'Monsoon', 7:'Monsoon', 8:'Monsoon', 9:'Monsoon',
                  10:'Post-Monsoon', 11:'Post-Monsoon'}
    df['season'] = df['month'].map(season_map)

    for lag in [1, 2, 3, 12]:
        df[f"lag_{lag}"] = df["price"].shift(lag)

    df["rolling_mean_3"] = df["price"].shift(1).rolling(3).mean()
    df["rolling_std_3"] = df["price"].shift(1).rolling(3).std()
    df["diff_1"] = df["price"].diff()

    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    current_year = df.index.year.max()
    df["weight"] = np.exp(-0.3 * (current_year - df.index.year))

    df = df.dropna()

    return df

def feature_engineering(df):
    # ========================
    # 1. Set index
    # ========================
    df.set_index('date', inplace=True)
    
    # Ensure monthly frequency
    df = df.asfreq('MS')
    
    df['price']=df['avg_monthly_price']
    
    # =========================
    # 2. LOG TRANSFORM
    # =========================
    
    df['price'] = log_transform(df['price'])

    # =========================
    # 3. LAG FEATURES
    # =========================
    
    df['lag_1'] = df['price'].shift(1)
    df['lag_2'] = df['price'].shift(2)
    df['lag_3'] = df['price'].shift(3)
    df['lag_12'] = df['price'].shift(12)
    
    # =========================
    # 4. ROLLING FEATURES
    # =========================
    
    df['rolling_mean_3'] = df['price'].shift(1).rolling(3).mean()
    df['rolling_std_3'] = df['price'].shift(1).rolling(3).std()
    
    # =========================
    # 5. TREND FEATURES
    # =========================
    
    df['diff_1'] = df['price'].diff(1)
    
    # =========================
    # 6. SEASONAL FEATURES
    # =========================
    
    df['month'] = df.index.month
    
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # =========================
    # 7. YEARLY CONTEXT
    # =========================
    
    df['avg_price_last_year'] = df['price'].shift(1).rolling(12).mean()
    
    # =========================
    # 8. RECENCY WEIGHT
    # =========================
    
    current_year = df.index.year.max()
    df['weight'] = np.exp(-0.3 * (current_year - df.index.year))
    
    # =========================
    # 9. DROP UNUSED COLUMNS
    # =========================
    
    drop_cols = [
    'avg_monthly_price', 'month_name', 'quarter', 'season',
    'iqr_outlier', 'zscore', 'zscore_outlier',
    'roll_3', 'roll_6', 'roll_12', 'ewm_24',
    'price_diff', 'pct_change',
    'rolling_std3', 'rolling_std6'
    ]
    
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    return df