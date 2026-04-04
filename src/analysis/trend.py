import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker

PALETTE = sns.color_palette('tab10')
FMT = mticker.FuncFormatter(lambda x, _: f'{x:,.0f}')

def trend_plots(df):
    df['roll_3']  = df['avg_monthly_price'].rolling(window=3,  center=True).mean()
    df['roll_6']  = df['avg_monthly_price'].rolling(window=6,  center=True).mean()
    df['roll_12'] = df['avg_monthly_price'].rolling(window=12, center=True).mean()
    df['ewm_24']  = df['avg_monthly_price'].ewm(span=24, adjust=False).mean()
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    axes[0].plot(df['date'], df['avg_monthly_price'], color='lightsteelblue', lw=1, alpha=0.8, label='Original')
    axes[0].plot(df['date'], df['roll_3'], color=PALETTE[1], lw=2, label='3-month MA')
    axes[0].plot(df['date'], df['roll_6'], color=PALETTE[2], lw=2, label='6-month MA')
    axes[0].set_title('Short-term Trend (3 & 6 month Rolling Mean)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Price'); axes[0].yaxis.set_major_formatter(FMT); axes[0].legend()
    
    axes[1].plot(df['date'], df['avg_monthly_price'], color='lightsteelblue', lw=1, alpha=0.8, label='Original')
    axes[1].plot(df['date'], df['roll_12'], color=PALETTE[3], lw=2.5, label='12-month MA')
    axes[1].plot(df['date'], df['ewm_24'],  color=PALETTE[4], lw=2.5, ls='--', label='EWM span=24')
    axes[1].set_title('Long-term Trend (12-month MA & EWM)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Date'); axes[1].set_ylabel('Price'); axes[1].yaxis.set_major_formatter(FMT); axes[1].legend()
    plt.tight_layout()
    return (fig, df)