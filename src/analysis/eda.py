import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

PALETTE = sns.color_palette('tab10')
FMT = mticker.FuncFormatter(lambda x, _: f'{x:,.0f}')

def overview(df):
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    shape = df.shape
    Daterange = f'{df.date.min().date()} to {df.date.max().date()}'
    months_covered = df.shape[0]
    years_covered = df.date.dt.year.nunique()
    columns = df.columns

    stats_s = df['avg_monthly_price'].describe().round(2)
    stats = stats_s.to_string()
    mean_ = df['avg_monthly_price'].mean()
    std_  = df['avg_monthly_price'].std()
    CV = f'{std_/mean_*100:.1f}%'
    skewness = f"{df.avg_monthly_price.skew():.3f}"
    kurtosis = f"{df.avg_monthly_price.kurtosis():.3f}"

    reports = {
        "Overview" : {
            "Shape" : shape,
            "Daterange" : Daterange,
            "Months covered" : months_covered,
            "Years covered" : years_covered,
            "Columns" : columns,
            "Stats" : stats,
            "CV" : CV,
            "Skewness" : skewness,
            "Kurtosis" : kurtosis
        }
    }

    return (reports,df)

def overview_plot(df):
    fig, ax = plt.subplots(figsize=(15, 4))
    
    ax.plot(df['date'], df['avg_monthly_price'], color=PALETTE[0], lw=1.5)
    ax.fill_between(df['date'], df['avg_monthly_price'], alpha=0.1, color=PALETTE[0])
    
    ax.set_title('Average Monthly Price - Full Series', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.yaxis.set_major_formatter(FMT)
    
    plt.tight_layout()
    return (fig,df)

def Time_Series_Structure_Check(df):
    reports={}
    adf = adfuller(df['avg_monthly_price'], autolag='AIC')
    reports['Augmented Dickey-Fuller Test']={ 
        'ADF Statistic': f"{adf[0]:.4f}",
        'p-value': f"{adf[1]:.4f}"
    }
    for k, v in adf[4].items():
        reports['Augmented Dickey-Fuller Test'][f'Critical {k}'] = f"{v:.4f}"
    conclusion = 'STATIONARY' if adf[1] < 0.05 else 'NON-STATIONARY'
    reports['Augmented Dickey-Fuller Test']['Conclusion'] = conclusion

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(df['avg_monthly_price'], lags=36, ax=axes[0], color=PALETTE[0])
    axes[0].set_title('ACF', fontweight='bold')
    plot_pacf(df['avg_monthly_price'], lags=36, ax=axes[1], color=PALETTE[1])
    axes[1].set_title('PACF', fontweight='bold')
    plt.suptitle('ACF & PACF - Identifying Lag Structure for SARIMA', fontsize=12, y=1.02)
    plt.tight_layout()

    return (reports['Augmented Dickey-Fuller Test'],fig,df)

def Month_Wise_Analysis(df):
    fig1, axes1 = plt.subplots(6, 2, figsize=(14, 20), sharex=True, sharey=True)
    axes1 = axes1.flatten()
    
    colors = plt.cm.tab20(np.linspace(0, 1, 12))
    pivot = df.pivot_table(index='year', columns='month',
                       values='avg_monthly_price', aggfunc='mean')
    # Yearly average across all months
    MONTH_LABELS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    pivot.columns = MONTH_LABELS
    year_avg = pivot[MONTH_LABELS].mean(axis=1)
    
    for i, month in enumerate(MONTH_LABELS):
        if month in pivot.columns:
            ax = axes1[i]
            
            # Monthly trend
            ax.plot(pivot.index, pivot[month],
                    marker='o', ms=4,
                    color=colors[i],
                    lw=1.5,
                    label=month)
            
            # Yearly average trend
            ax.plot(pivot.index, year_avg,
                    linestyle='--',
                    color='black',
                    lw=2,
                    label='Year Avg')
            
            ax.set_title(month, fontsize=11, fontweight='bold')
            ax.grid(True)
    
    # Common labels
    fig1.suptitle('Month-wise vs Yearly Average Price Trend', fontsize=16, fontweight='bold')
    
    # Shared legend
    handles, labels = axes1[0].get_legend_handles_labels()
    fig1.legend(handles, labels, loc='upper right', ncol=1, fontsize=9)
    
    plt.tight_layout()

    fig2, ax = plt.subplots(figsize=(14, 5))
    month_data = [df[df['month'] == m+1]['avg_monthly_price'].values for m in range(12)]
    bp = ax.boxplot(month_data, labels=MONTH_LABELS, patch_artist=True,
                    medianprops=dict(color='black', lw=2))
    cmap = plt.cm.RdYlGn(np.linspace(0.15, 0.85, 12))
    for patch, c in zip(bp['boxes'], cmap):
        patch.set_facecolor(c)
    ax.set_title('Price Distribution by Month', fontsize=13, fontweight='bold')
    ax.set_xlabel('Month'); ax.set_ylabel('Price'); ax.yaxis.set_major_formatter(FMT)
    plt.tight_layout()

    return(fig1,fig2,df)

def Outlier(df):
    reports={}
    Q1  = df['avg_monthly_price'].quantile(0.25)
    Q3  = df['avg_monthly_price'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df['iqr_outlier'] = (df['avg_monthly_price'] < lower) | (df['avg_monthly_price'] > upper)
    
    reports['Outliers']={}
    reports['Outliers']['IQR Outlier Detection'] = (
    f"Q1={Q1:,.0f}, Q3={Q3:,.0f}, IQR={IQR:,.0f}, "
    f"Lower={lower:,.0f}, Upper={upper:,.0f}, "
    f"Outliers={df.iqr_outlier.sum()}"
)
    if df.iqr_outlier.sum():
        print(df[df.iqr_outlier][['date','year','month_name','avg_monthly_price']].to_string(index=False))

    df['zscore'] = np.abs(stats.zscore(df['avg_monthly_price']))
    df['zscore_outlier'] = df['zscore'] > 3
    reports['Outliers']['Z-score'] = f"Z-score outliers (|z|>3): {df.zscore_outlier.sum()}"
    if df.zscore_outlier.sum():
        a=(df[df.zscore_outlier][['date','year','month_name','avg_monthly_price','zscore']].to_string(index=False))
        reports['Outliers']['Z-score outliers'] = a
    

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].boxplot(df['avg_monthly_price'], vert=True, patch_artist=True,
                    boxprops=dict(facecolor='#74b9ff'), medianprops=dict(color='navy', lw=2),
                    flierprops=dict(marker='o', color='red', markersize=6))
    axes[0].axhline(upper, color='red', ls='--', lw=1, label=f'Upper fence {upper:,.0f}')
    axes[0].axhline(lower, color='orange', ls='--', lw=1, label=f'Lower fence {lower:,.0f}')
    axes[0].set_title('Overall Price Boxplot', fontweight='bold')
    axes[0].set_ylabel('Price'); axes[0].yaxis.set_major_formatter(FMT); axes[0].legend(fontsize=8)
    
    outliers_df = df[df['iqr_outlier']]
    axes[1].plot(df['date'], df['avg_monthly_price'], color='steelblue', lw=1.2, label='Price')
    axes[1].scatter(outliers_df['date'], outliers_df['avg_monthly_price'],
                    color='red', zorder=5, s=70, label=f'Outliers ({len(outliers_df)})')
    axes[1].axhline(upper, color='red', ls='--', lw=1, alpha=0.6)
    axes[1].axhline(lower, color='orange', ls='--', lw=1, alpha=0.6)
    axes[1].set_title('Anomalies on Time Series', fontweight='bold')
    axes[1].set_ylabel('Price'); axes[1].yaxis.set_major_formatter(FMT); axes[1].legend(fontsize=8)
    
    plt.tight_layout()

    return (reports['Outliers'],fig,df)