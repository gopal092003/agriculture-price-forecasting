import pandas as pd
from src.analysis.eda import overview, overview_plot, Time_Series_Structure_Check, Month_Wise_Analysis, Outlier
from src.analysis.seasonality import season_wise_data, season_wise_plots
from src.analysis.trend import trend_plots
from src.analysis.volatility import volatility_plots
from src.data.clean_data import clean_data
from src.data.validate_data import validate_data
from src.features.build_features import build_features
from src.utils.plotting import save_plot
from src.utils.save_outputs import save_csv, save_json
from src.config.config import load_config

import os
import json

def run():
    # Load config
    paths = load_config("config/paths_config.yaml")

    raw_path = paths["data"]["raw"]
    interim_path = paths["data"]["interim"]
    processed_path = paths["data"]["processed"]
    final_path = paths["data"]["final"]
    plots_base = paths["outputs"]["plots"]
    reports_base = paths["outputs"]["reports"]

    df = pd.read_csv(raw_path, parse_dates=["date"])

    df = clean_data(df)
    save_csv(df, interim_path)

    # -------------------------------
    # EDA
    # -------------------------------
    reports,df = overview(df)
    
    fig,df = overview_plot(df)
    save_plot(fig, f"{plots_base}/eda/Average Monthly Price - Full Series.png")

    MONTH_LABELS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    df = build_features(df)
    df = df.reset_index()
    save_csv(df, processed_path)

    reports['Augmented Dickey-Fuller Test'],fig2,df = Time_Series_Structure_Check(df)
    save_plot(fig2, f"{plots_base}/eda/ACF & PACF - Identifying Lag Structure for SARIMA.png")

    fig3, fig4,df = Month_Wise_Analysis(df)
    save_plot(fig3, f"{plots_base}/eda/Month-wise vs Yearly Average Price Trend.png")
    save_plot(fig4, f"{plots_base}/eda/Price Distribution by Month.png")

    # -------------------------------
    # Seasonality
    # -------------------------------
    reports['Season_wise'],season_yearly,SEASON_ORDER,SEASON_COLORS,df = season_wise_data(df)
    fig5, fig6, reports['Season-wise CV%'],df  = season_wise_plots(df,season_yearly,SEASON_ORDER,SEASON_COLORS)
    save_plot(fig5, f"{plots_base}/seasonality/Season-wise Price Trend Across Years.png")
    save_plot(fig6, f"{plots_base}/seasonality/Average Price by Season with Std Dev.png")

    # -------------------------------
    # Outliers
    # -------------------------------
    reports['Outliers'],fig7,df = Outlier(df)
    save_plot(fig7, f"{plots_base}/eda/Outliers.png")

    # -------------------------------
    # Trend
    # -------------------------------
    fig8,df = trend_plots(df)
    save_plot(fig8, f"{plots_base}/trend/Trend.png")

    # -------------------------------
    # Volatility
    # -------------------------------
    fig9,fig10, reports['Volatility'],df = volatility_plots(df)
    save_plot(fig9, f"{plots_base}/volatility/Absolute MoM Price Change.png")
    save_plot(fig10, f"{plots_base}/volatility/Price Volatility by Year.png")

    save_csv(df, final_path)

    file_path = os.path.join(reports_base, "reports.txt")
    
    with open(file_path, "w", encoding="utf-8") as f:
        for item in reports:
            f.write(f"{item}\n")